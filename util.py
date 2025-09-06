from transformers import StoppingCriteria
import torch
import time
import re

def extract_steps_meta(text):
    """
    Parse <step n="..." ref="...">...</step> tags.
    Returns list of dicts: {"n":int, "ref":List[int]|["p"], "text":str}
    """
    pattern = re.compile(
        r'<step\s+n="(?P<n>\d+)"\s+ref="(?P<ref>[^"]+)"\s*>(?P<text>.*?)</step>',
        re.DOTALL
    )
    matches = pattern.finditer(text)

    if not matches:
        return [{"n": i + 1, "ref": "p", "text": s} for i, s in enumerate(extract_steps(text))]
    
    steps = []
    for m in matches:
        raw_ref = m.group("ref")
        ref = [x for x in raw_ref.split(",")]
        steps.append({
            "n": int(m.group("n")),
            "ref": ref,
            "text": m.group("text").strip()
        })
    return steps

def extract_steps(text):
    pattern = re.compile(
        r'<step>(?P<text>.*?)</step>',
        re.DOTALL
    )
    steps = []
    for m in pattern.finditer(text):
        steps.append(m.group("text").strip())
    return steps

def extract_answer_meta(text):
    """
    Parse <answer ref="...">...</answer>.
    Returns string answer
    """
    m = re.search(
        r'<answer\s+ref="(?P<ref>[^"]+)"\s*>(?P<text>.*?)</answer>',
        text, re.DOTALL
    )
    if not m:
        return extract_answer(text)
    return m.group("text").strip()

def extract_answer(text):
    m = re.search(
        r'<answer>(?P<text>.*?)</answer>',
        text, re.DOTALL
    )
    if not m:
        return None
    return m.group("text").strip()

class AnswerEOS(StoppingCriteria):
    def __init__(self, tokenizer):
        self.batch_size = -1
        self.seen = []
        self.total = 0
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs):
        if self.batch_size == -1:
            self.batch_size = input_ids.shape[0]
            self.seen = [False] * self.batch_size
        for i in range(self.batch_size):
            if self.seen[i]: continue
            txt = self.tokenizer.decode(input_ids[i, -10:])
            if "</answer>" in txt:
                self.seen[i] = True
                self.total += 1
        return self.total == self.batch_size

default_args = {"max_new_tokens": 2000, "do_sample": True, "temperature": 0.5}
@torch.no_grad()
def prompt_model_answer(prompts: list[str], model, tokenizer, *, debug: bool = False, **model_args):
    start = time.time()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to("cuda")
    end = time.time()
    if debug: print(f"tokenized inputs in {(end - start):0.3f}s")

    start = time.time()
    full_args = default_args | model_args
    outputs = model.generate(
        **inputs,
        **full_args,
        stopping_criteria=[AnswerEOS(tokenizer)],
        eos_token_id=None
    )
    end = time.time()
    if debug: print(f"generated outputs in {(end - start):0.3f}s")

    start = time.time()
    result = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    end = time.time()
    if debug: print(f"decoded outputs in {(end - start):0.3f}s")
    
    def extract(res):
        try:
            return res.split("<answer>")[1].split("</answer>")[0].strip()
        except:
            return "ERROR"
    return [(r, extract(r)) for r in result]