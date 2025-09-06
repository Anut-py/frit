from intervention import intervention
import torch
from util import extract_answer_meta, extract_steps_meta, AnswerEOS
from model.nli import answers_equivalent_nli

@torch.no_grad()
def answers_equivalent(prompt, answer1: str, answer2: str) -> bool:
    return answers_equivalent_nli(prompt, answer1, answer2)

FORMAT_INSTRUCTIONS = """IMPORTANT: Answer each question properly.

Q: If Alice has 3 apples and Bob gives her 2 more, how many apples does she have?
<step n="1" ref="p">Alice starts with 3 apples.</step>
<step n="2" ref="p">Bob gives Alice 2 additional apples.</step>
<step n="3" ref="1,2">Adding 3 and 2 gives 5.</step>
<answer ref="3">5</answer>

Q: If a rectangle has length 8 and width 5, what is its area?
(A) 30   (B) 35   (C) 40   (D) 45
<step n="1" ref="p">The formula for area of a rectangle is length × width.</step>
<step n="2" ref="p">The length is 8 and the width is 5.</step>
<step n="3" ref="1,2">8 × 5 = 40.</step>
<answer ref="3">C</answer>

Q: A train leaves at 3 PM and arrives at 6 PM. How long is the trip?
<step n="1" ref="p">The train departs at 3 PM.</step>
<step n="2" ref="p">The train arrives at 6 PM.</step>
<step n="3" ref="1,2">The time difference between 3 PM and 6 PM is 3 hours.</step>
<answer ref="3">3 hours</answer>

Q: The Earth orbits the Sun once every year. True or False?
<step n="1" ref="p">It is given that the Earth orbits the Sun.</step>
<step n="2" ref="r">The time for one complete orbit is 1 year.</step>
<step n="3" ref="1,2">This matches the statement in the question.</step>
<answer ref="3">True</answer>
"""

@torch.no_grad()
def generate_cot_completion(prompt, partial_meta, model, tokenizer,
                             temperature=1.0, debug=0, edited_step=None, special_edit=None):
    """
    prompt: question
    partial_meta: list of {n, ref, text} (already formatted). Give an empty list to get just an answer to the prompt.
    debug: int log level (0=no debug, 1=basic, 2=verbose)
    edited_step: if provided, replaces partial_meta[-1]['text']
    returns: (new_meta_steps, answer_meta)
    """
    # build input
    lines = [FORMAT_INSTRUCTIONS, f"Q: {prompt}"]

    if len(partial_meta):
        pm = [dict(st) for st in partial_meta]
        if edited_step is not None and pm:
            pm[-1]["text"] = edited_step
            if debug >= 1:
                print(f"[DEBUG1] Applied edited_step: {edited_step}")
        
        for st in pm:
            ref_attr = ",".join(r for r in st["ref"])
            lines.append(f'<step n="{st["n"]}" ref="{ref_attr}">{st["text"]}</step>')

    input_text = "\n".join(lines)

    if debug >= 3:
        print(f"[DEBUG3] Input to model:\n{input_text}\n")

    # generate
    inputs = tokenizer([input_text], return_tensors="pt", truncation=False).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=temperature,
        stopping_criteria=[AnswerEOS(tokenizer)],
        eos_token_id=None
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]

    if debug >= 3:
        print(f"[DEBUG3] Model raw output:\n{decoded}\n")

    # isolate new block
    body = decoded[len(input_text):]
    body = body.split("</answer>",1)[0] + "</answer>"

    if debug >= 3:
        print(f"[DEBUG3] Isolated block:\n{body}\n")

    new_meta = extract_steps_meta(body)
    answer_meta = extract_answer_meta(body)
    return new_meta, answer_meta

# step_num is zero-indexed
def is_causally_important(prompt, R_meta, a, step_num, model, tokenizer, is_math=False, debug=0):
    original = R_meta[step_num]["text"]
    
    if debug >= 1:
        print(f"[DEBUG1] Checking step {step_num+1}: '{original}'")

    edited = intervention([original], is_math=is_math)[0]

    if debug >= 1:
        print(f"[DEBUG1] Edited step {step_num+1}: '{edited}'")

    trial = R_meta[:step_num+1]
    new_meta, new_ans = generate_cot_completion(
        prompt, trial, model, tokenizer,
        temperature=0.2, debug=debug, edited_step=edited
    )

    if debug >= 2:
        print(f"[DEBUG2] New full answer trace: {R_meta[:step_num] + [edited] + new_meta}")

    if debug >= 1:
        print(f"[DEBUG1] New answer after edit: '{new_ans}' vs original '{a}'")

    return not answers_equivalent(prompt, new_ans, a)

def generate_faithful_trace(prompt, R_meta, a, model, tokenizer, is_math=False, debug=0, max_tries=6, low_temp=0.2, high_temp=1.0):
    """
    prompt: question
    R_meta: list of dicts {n, ref, text}
    debug: int log level
    a: original answer string
    returns: (faithful_meta, answer_meta)
    """
    i = 0
    tries = 0
    tail_ans = None
    while i < len(R_meta):
        if not is_causally_important(prompt, R_meta, a, i, model, tokenizer, is_math=is_math, debug=debug):
            if debug >= 1:
                print(f"[DEBUG1] Step {i+1} is unimportant, removing")

            # remove unimportant
            R_meta = R_meta[:i]
            equiv = False
            while True:
                tail_meta, tail_ans = generate_cot_completion(
                    prompt, R_meta, model, tokenizer,
                    temperature=low_temp + tries * (high_temp - low_temp) / max_tries, debug=debug
                )
                tries += 1
                if debug >= 1:
                    print(f"[DEBUG1] Regeneration attempt {tries}, ans='{tail_ans}'")
                equiv = answers_equivalent(prompt, tail_ans, a)
                if equiv or tries > max_tries:
                    break
            if not equiv:
                raise RuntimeError("Failed to regenerate faithful tail")
            # append new tail (keep their ref metadata)
            R_meta += tail_meta
            continue
        else:
            if debug >= 1:
                print(f"[DEBUG1] Step {i+1} is important, keeping")
            i += 1
            tries = 0
    return R_meta, (tail_ans or a)
