import sys
from pathlib import Path

# add the parent (or any) directory to sys.path
parent = Path().resolve().parent  # one level up
sys.path.insert(0, str(parent))

import pickle

representatives_file_name = "data/representatives.pkl"

with open(representatives_file_name, "rb") as representatives_file:
    representatives = pickle.load(representatives_file)

from model.model import load_tokenizer, load_base_model
from util import prompt_model_answer

tokenizer = load_tokenizer(True)
base_model = load_base_model(True)

from tqdm import tqdm

label_prompt_template = """You are given a list of representative facts, all from the same domain.  
Your task is to:

1. Examine the facts and identify their common theme.  
2. Reason step by step—list key observations about what these facts have in common.  
3. Determine the most concise, descriptive label for this domain (2–4 words).  

**Important**:  
- Do **not** include anything besides your reasoning steps and the final label.  
- Wrap **only** the final label in `<answer>...</answer>` tags, and nothing else.

Facts:
%s

Step-by-step reasoning:"""

batch_size = 10
cluster_ids = list(sorted(list(representatives.keys())))

full_responses_file_name = "data/full_label_resps.pkl"
cluster_labels_file_name = "data/cluster_labels.pkl"

cluster_labels = dict()
full_resps = dict()

cluster_batches = [cluster_ids[i : i + batch_size] for i in range(0, len(cluster_ids), batch_size)]
for cluster_batch in tqdm(cluster_batches):
    fact_batch = [representatives[i] for i in cluster_batch]
    prompts = [label_prompt_template % ("\n".join(f"- {fact}" for fact in rep_facts)) for rep_facts in fact_batch]
    answers = prompt_model_answer(prompts, 2000, base_model, tokenizer, False)
    for i, cluster in enumerate(cluster_batch):
        full, answer = answers[i]
        full_resps[cluster] = full
        cluster_labels[cluster] = answer

    with open(full_responses_file_name, "wb") as full_responses_file:
        pickle.dump(full_resps, full_responses_file)

    with open(cluster_labels_file_name, "wb") as cluster_labels_file:
        pickle.dump(cluster_labels, cluster_labels_file)

print("done")
