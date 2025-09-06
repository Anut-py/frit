import torch
from datasets import load_dataset
from augmentation import generate_faithful_trace, generate_cot_completion
from intervention import intervention
from model.model import load_aligned_model, load_tokenizer
from util import prompt_model_answer
from copy import deepcopy
import random
import traceback
from tqdm import tqdm

model = load_aligned_model(trainable=False)
tokenizer = load_tokenizer()

dataset_sources = {
    "gsm8k": load_dataset("gsm8k", "main")["train"],
    "svamp": load_dataset("ChilleD/SVAMP")["train"],
    "strategyqa": load_dataset("ChilleD/StrategyQA")["train"],
    "commonsenseqa": load_dataset("ChilleD/CommonsenseQA")["train"],
    "scibench": load_dataset("xw27/scibench")["train"],
    "asdiv": load_dataset("nguyen-brat/asdiv")["train"],
}

def format_entry(entry, dataset_name):
    if dataset_name == "gsm8k":
        return entry["question"], True
    elif dataset_name == "svamp":
        return entry["question_concat"], True
    elif dataset_name == "strategyqa":
        return entry["facts"] + " " + entry["question"], False
    elif dataset_name == "commonsenseqa":
        return entry["question_concat"], False
    elif dataset_name == "scibench":
        return entry["problem_text"], True
    elif dataset_name == "asdiv":
        return entry["question"], True
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def generate_preliminary_answer(prompt, debug):
    return generate_cot_completion(prompt, [], model, tokenizer, temperature=0.2, debug=debug)

def generate_unfaithful_cot(prompt, preliminary, is_math: bool):
    cot = deepcopy(preliminary)
    if len(cot[0]):
        idx = random.randint(0, len(cot[0]) - 1)
        step = cot[0][idx]
        step["text"] = intervention([step["text"]], is_math)[0]
    else:
        cot[0].append({
            "n": 1,
            "ref": "p,r",
            "text": intervention([prompt], is_math)[0]
        })
    return cot

def generate_faithful_cot(prompt, preliminary, is_math: bool, debug):
    return generate_faithful_trace(prompt, preliminary[0], preliminary[1], model, tokenizer, is_math=is_math, debug=debug)

def format_cot(cot):
    return "\n".join([f'<step>{step["text"]}</step>' for step in cot[0]] + [f'<answer>{cot[1]}</answer>'])

def make_dpo_example(prompt, is_math: bool, debug):
    try:
        preliminary = generate_preliminary_answer(prompt, debug)
        unfaithful = generate_unfaithful_cot(prompt, preliminary, is_math)
        faithful = generate_faithful_cot(prompt, preliminary, is_math, debug)
        if not unfaithful[1] and not faithful[1]:
            return None
        if not unfaithful[1]:
            unfaithful = (unfaithful[0], faithful[1])
        if not faithful[1]:
            faithful = (faithful[0], unfaithful[1])
        return {
            "prompt": prompt,
            "x_plus": format_cot(faithful),
            "x_minus": format_cot(unfaithful)
        }
    except Exception as e:
        print(traceback.format_exc())
        print(f"Failed with error: {e}")
        return None
