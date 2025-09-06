#!/usr/bin/env python3
import builtins
print = lambda *args, **kwargs: builtins.print(*args, **{**kwargs, "flush": True})

import os
import pickle
import time
import random
import fcntl
from multiprocessing import Process, set_start_method
from tqdm import tqdm
import json
import math
import torch, gc
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
import torch
from config import OUT_DIR, data_subdir, DPO_CONFIG, GEN_EPOCHS as EPOCHS

torch.set_grad_enabled(True)

from model.model import load_tokenizer, load_base_model, load_aligned_model, save_aligned_model

tokenizer = load_tokenizer()
ref_model = load_base_model()
model = load_aligned_model()

model.print_trainable_parameters()

def clear_cuda():
    gc.collect()                 # clear Python garbage
    torch.cuda.empty_cache()     # release cached blocks to the driver
    torch.cuda.synchronize()     # wait for all pending CUDA ops to finish

EPOCHS = 3
PICKLE_PATH = data_subdir + "/datagen%d.pkl"
STATUS_PATH = OUT_DIR + "/dpo_status.json"

dpo_cfg = DPO_CONFIG
trainer = None

def run_dpo(epoch):
    global trainer, model, ref_model, tokenizer, dpo_cfg
    
    with open(PICKLE_PATH % epoch, "rb") as f:
        dpo_triples = pickle.load(f)

    preference_dataset = Dataset.from_list([{"prompt": t["prompt"], "chosen": t["x_plus"], "rejected": t["x_minus"]} for t in dpo_triples])

    def tokenize_dpo(example):
        global tokenizer
        prompt = tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        chosen = tokenizer(example["chosen"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        rejected = tokenizer(example["rejected"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        return {
            "prompt_input_ids": prompt.input_ids.squeeze(0),
            "chosen_input_ids": chosen.input_ids.squeeze(0),
            "rejected_input_ids": rejected.input_ids.squeeze(0),
        }
    
    preference_dataset = preference_dataset.map(lambda x: tokenize_dpo(x))

    if trainer:
        trainer.train_dataset = preference_dataset
    else:
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            processing_class=tokenizer,
            args=DPOConfig(**dpo_cfg),
            train_dataset = preference_dataset
        )

    dpo_status = {"state": "running", "start_time": time.time(), "gen_epoch": epoch}
    with open(STATUS_PATH, "w") as sf:
        json.dump(dpo_status, sf)
        sf.flush()
        os.fsync(sf.fileno())

    result = trainer.train()

    dpo_status = {
        "state": "finished",
        "end_time": time.time(),
        "gen_epoch": epoch,
        "metrics": getattr(result, "metrics", None) or {}
    }
    with open(STATUS_PATH, "w") as sf:
        json.dump(dpo_status, sf)
    
    save_aligned_model(model)

# ---------- main ----------
if __name__ == "__main__":
    for epoch in range(EPOCHS):
        print(f"=== EPOCH {epoch} ===")
        clear_cuda()
    
        # remove any leftover per-epoch files so generation starts fresh
        try:
            os.remove(STATUS_PATH)
        except FileNotFoundError:
            pass

        # run DPO on the epoch file (pass gen_epoch so run_dpo can write status)
    
        print(f"[Main] starting DPO for epoch {epoch}")
        # pass gen_epoch to allow status writes
        run_dpo(epoch)
        try:
            with open(STATUS_PATH, "r") as sf:
                s = json.load(sf)
                print("[Main] DPO metrics:", s.get("metrics"))
        except Exception:
            pass
        print(f"[Main] finished DPO for epoch {epoch}")

        clear_cuda()
    
    print("All epochs complete.")
