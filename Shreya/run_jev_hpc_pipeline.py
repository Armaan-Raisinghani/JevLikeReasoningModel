#!/usr/bin/env python3
from __future__ import annotations
"""All-in-One Standalone Jev Escalation Pipeline for SLURM / HPC Clusters.

This script executes the complete end-to-end pipeline:
  1. Data Preparation: Downloads or loads canonical dataset splits (SST-2, ANLI,
     ARC-Challenge, MMLU, and held-out AQUA-RAT math) with leakage assertions.
  2. Teacher Traces: Rejection-samples reasoning traces from training data.
  3. Training (LoRA SFT): Trains a parameter-efficient adapter using the custom
     ReadoutTrainer (joint language modeling loss on reasoning tokens +
     restricted-option cross-entropy readout loss on answer tokens).
  4. Validation Calibration: Fits the confidence escalation gate threshold
     on validation data within a specified target accuracy gap.
  5. Test Analysis: Executes online gated inference and computes latency, forward
     passes, generated tokens, calibration metrics (ECE, Brier, NLL), per-task
     breakdowns, and exports publication-ready tables and JSON metrics.

Usage Examples:
  # 1. Quick smoke test (verify GPU, dependencies, and pipeline end-to-end in ~5 mins):
  python run_jev_hpc_pipeline.py --root /scratch/$USER/jev_exp --smoke-test

  # 2. Full training and evaluation run on HPC:
  python run_jev_hpc_pipeline.py --root /scratch/$USER/jev_exp --base-model Qwen/Qwen3-1.7B

  # 3. Run only training:
  python run_jev_hpc_pipeline.py --root /scratch/$USER/jev_exp --train-only

  # 4. Run only analysis (if checkpoint already trained):
  python run_jev_hpc_pipeline.py --root /scratch/$USER/jev_exp --eval-only
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ==============================================================================
# Task Definitions and Formatting
# ==============================================================================

TASKS = ["sentiment", "inference", "science questions", "general knowledge", "word math"]
DESCRIPTIONS = {
    "sentiment": "Classify the sentiment of the review as positive or negative.",
    "inference": "Determine whether the hypothesis is entailed, contradicted, or neutral given the premise.",
    "science questions": "Answer the grade school science question.",
    "general knowledge": "Answer the multiple-choice exam question.",
    "word math": "Solve the math word problem.",
}
LETTERS = list("ABCDE")

HF_DATASETS = {
    "sentiment": ("stanfordnlp/sst2", None),
    "inference": ("facebook/anli", None),
    "science questions": ("allenai/ai2_arc", "ARC-Challenge"),
    "general knowledge": ("cais/mmlu", "all"),
    "word math": ("deepmind/aqua_rat", "raw"),
}

HF_SOURCE_SPLITS = {
    "sentiment": ("train", "validation", "validation"),
    "inference": ("train_r1", "dev_r1", "test_r1"),
    "science questions": ("train", "validation", "test"),
    "general knowledge": ("auxiliary_train", "dev", "test"),
    "word math": ("train", "validation", "test"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Execution Paths
    parser.add_argument("--root", type=Path, default=Path("./jev_experiment"),
                        help="Root directory for data, checkpoints, and metrics.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-1.7B",
                        help="Base model identifier or path.")
    parser.add_argument("--teacher-model", type=str, default=None,
                        help="Teacher model for traces (default: uses base-model).")
    parser.add_argument("--adapter-name", type=str, default="lora_adapter",
                        help="Subdirectory name inside checkpoints/.")

    # Execution Modes
    parser.add_argument("--train-only", action="store_true", help="Run only data prep & training.")
    parser.add_argument("--eval-only", action="store_true", help="Run only analysis & evaluation.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run small test (20 samples per task, 1 epoch) to verify pipeline.")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Retrain adapter even if checkpoint exists.")
    parser.add_argument("--force-redownload", action="store_true",
                        help="Rebuild splits.json from Hugging Face even if file exists.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Evaluation Hyperparameters
    parser.add_argument("--max-reason-tokens", type=int, default=256)
    parser.add_argument("--target-accuracy-gap", type=float, default=0.03,
                        help="Acceptable accuracy gap between reasoning-only and gated policy.")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup examples excluded from latency profiling.")
    parser.add_argument("--seed", type=int, default=2026)

    return parser.parse_args()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prompt_for(ex: Dict[str, Any], reasoning: bool = False) -> str:
    body = f"Task: {DESCRIPTIONS[ex['task']]}\nQuestion: {ex['question']}\nOptions:\n"
    body += "".join(f"{LETTERS[i]}. {opt}\n" for i, opt in enumerate(ex["options"]))
    body += "Answer:"
    if reasoning:
        body += "\nReasoning:"
    return body


# ==============================================================================
# 1. Dataset Loading, Canonicalization, and Leakage Guards
# ==============================================================================

def norm_text(val: Any) -> str:
    return str(val).strip()


def letters_to_label(answer: Any, letters: List[str]) -> int | None:
    a = str(answer).strip().upper().replace("(", "").replace(")", "").replace(".", "")
    if a in letters:
        return letters.index(a)
    try:
        return int(a)
    except Exception:
        return None


def map_row(task: str, row: Dict[str, Any]) -> Tuple[str, List[str], int | None]:
    if task == "sentiment":
        return "Review: " + norm_text(row["sentence"]), ["negative", "positive"], int(row["label"])
    if task == "inference":
        return (
            "Premise: " + norm_text(row["premise"]) + "\nHypothesis: " + norm_text(row["hypothesis"]),
            ["entailment", "neutral", "contradiction"],
            int(row["label"]),
        )
    if task == "science questions":
        ch = row["choices"]
        opts = [norm_text(x) for x in ch["text"]]
        labs = [str(x) for x in ch["label"]]
        return norm_text(row["question"]), opts, labs.index(str(row["answerKey"]))
    if task == "general knowledge":
        opts = [norm_text(x) for x in row["choices"]]
        return norm_text(row["question"]), opts, int(row["answer"])
    if task == "word math":
        opts = [norm_text(x) for x in row["options"]]
        labels = [x.split(")", 1)[0].strip(" (") for x in opts]
        opts = [x.split(")", 1)[1].strip() if ")" in x else x for x in opts]
        return norm_text(row["question"]), opts, letters_to_label(row["correct"], labels)
    raise KeyError(f"Unknown task: {task}")


def prepare_data_splits(args: argparse.Namespace) -> Dict[str, Dict[str, List[dict]]]:
    split_file = args.root / "data" / "splits.json"
    if split_file.exists() and not args.force_redownload:
        print(f"[Data] Loading existing splits from {split_file}", flush=True)
        return json.loads(split_file.read_text(encoding="utf-8"))

    print("[Data] Generating canonical splits from Hugging Face datasets...", flush=True)
    train_per_task, val_per_task, test_per_task = (
        (20, 20, 20) if args.smoke_test else (2000, 300, 500)
    )

    raw_datasets = {}
    ds_cache = str(args.root / "cache" / "datasets")
    for task, (ds_id, config) in HF_DATASETS.items():
        print(f"  Downloading/loading dataset: {task} ({ds_id})...", flush=True)
        raw_datasets[task] = load_dataset(ds_id, config, cache_dir=ds_cache) if config else load_dataset(ds_id, cache_dir=ds_cache)

    data: Dict[str, Dict[str, List[dict]]] = {}

    def take(ds, n, local_seed):
        ids = list(range(len(ds)))
        random.Random(local_seed).shuffle(ids)
        return [ds[i] for i in ids[:n]]

    def make_record(task, r):
        q, opts, label = map_row(task, r)
        if not opts or label is None or label < 0 or label >= len(opts):
            return None
        seed = int(hashlib.sha256((str(args.seed) + task + q).encode()).hexdigest()[:8], 16)
        order = list(range(len(opts)))
        random.Random(seed).shuffle(order)
        new_label = order.index(label)
        opts = [opts[i] for i in order]
        rec_id = hashlib.sha256(
            (task + "\n" + q + "\n" + "\n".join(opts) + "\n" + str(new_label)).encode()
        ).hexdigest()
        return {
            "id": rec_id,
            "task": task,
            "question": q,
            "options": opts,
            "label": new_label,
        }

    for task, ds in raw_datasets.items():
        tr, va, te = HF_SOURCE_SPLITS[task]
        if task == "sentiment":
            pool = ds["validation"]
            indices = list(range(len(pool)))
            random.Random(args.seed).shuffle(indices)
            nv = min(val_per_task, len(indices) // 2)
            val_raw = [pool[i] for i in indices[:nv]]
            test_raw = [pool[i] for i in indices[nv : nv + min(test_per_task, len(indices) - nv)]]
            train_raw = take(ds["train"], train_per_task, args.seed)
        else:
            train_raw = take(ds[tr], train_per_task, args.seed)
            val_raw = take(ds[va], val_per_task, args.seed + 1)
            test_raw = take(ds[te], test_per_task, args.seed + 2)

        def map_rows(rows):
            return [x for x in (make_record(task, r) for r in rows) if x is not None]

        data[task] = {
            "train": map_rows(train_raw),
            "val": map_rows(val_raw),
            "test": map_rows(test_raw),
        }
        print(f"  {task} counts: train={len(data[task]['train'])}, val={len(data[task]['val'])}, test={len(data[task]['test'])}", flush=True)

    # Rigorous leakage assertions
    train_ids = {x["id"] for t in TASKS[:-1] for x in data[t]["train"]}
    val_ids = {x["id"] for t in TASKS for x in data[t]["val"]}
    test_ids = {x["id"] for t in TASKS for x in data[t]["test"]}
    assert not (train_ids & val_ids or train_ids & test_ids), "ERROR: Split leakage detected!"
    # Ensure held-out word math is never in training
    assert not any(x["task"] == "word math" for t in TASKS[:-1] for x in data[t]["train"])

    (args.root / "data").mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[Data] Splits successfully materialized and saved to {split_file}", flush=True)
    return data


# ==============================================================================
# 2. Model & Tokenizer Loader
# ==============================================================================

def get_letter_tokens(tokenizer: Any) -> Dict[str, int]:
    letter_ids = {}
    for letter in LETTERS:
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(ids) != 1:
            ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Answer label {letter!r} does not map to a single token in tokenizer: {ids}")
        letter_ids[letter] = ids[0]
    return letter_ids


def load_model_and_tokenizer(
    model_name: str, adapter_path: Path | None = None
) -> Tuple[Any, Any, Dict[str, int], str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
        torch.float16 if device == "cuda" else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Model] Loading {model_name} on {device} (dtype={dtype})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if adapter_path is not None and (adapter_path / "adapter_config.json").exists():
        print(f"[Model] Attaching LoRA adapter from {adapter_path}...", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    if device == "cpu":
        model.to(device)

    letter_ids = get_letter_tokens(tokenizer)
    return model, tokenizer, letter_ids, device


# ==============================================================================
# 3. Teacher Reasoning Traces Generation
# ==============================================================================

def generate_teacher_traces(
    args: argparse.Namespace, data: Dict[str, Dict[str, List[dict]]]
) -> List[dict]:
    trace_path = args.root / "traces" / "train_traces.json"
    trace_total = 40 if args.smoke_test else 1500

    if trace_path.exists():
        existing = json.loads(trace_path.read_text(encoding="utf-8"))
        if len(existing) >= trace_total:
            print(f"[Traces] Using {len(existing)} existing teacher traces from {trace_path}", flush=True)
            return existing

    print(f"[Traces] Generating {trace_total} teacher reasoning traces...", flush=True)
    teacher_name = args.teacher_model or args.base_model
    teacher, tokenizer, _, device = load_model_and_tokenizer(teacher_name)

    eligible = [x for t in TASKS[:-1] for x in data[t]["train"]]
    random.Random(args.seed).shuffle(eligible)
    eligible = eligible[: min(len(eligible), trace_total * 2)]

    traces = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else []
    done_ids = {t["id"] for t in traces}

    for ex in eligible:
        if len(traces) >= trace_total:
            break
        if ex["id"] in done_ids:
            continue

        ask = prompt_for(ex) + "\nReason in under 150 words. End with exactly: Final answer: X"
        enc = tokenizer(ask, return_tensors="pt").to(next(teacher.parameters()).device)
        with torch.no_grad():
            gen = teacher.generate(**enc, max_new_tokens=220, do_sample=False, use_cache=True)
        completion = tokenizer.decode(gen[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)

        matches = re.findall(r"Final answer:\s*([ABCDE])", completion, re.I)
        if matches and LETTERS.index(matches[-1].upper()) == ex["label"]:
            reasoning = completion[: completion.lower().rfind("final answer:")].strip()
            traces.append({
                "id": ex["id"],
                "task": ex["task"],
                "prompt": prompt_for(ex),
                "reasoning": reasoning,
                "answer": LETTERS[ex["label"]],
            })
        done_ids.add(ex["id"])

        if len(traces) % 10 == 0 or len(traces) == trace_total:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
            print(f"  Traces accepted: {len(traces)}/{trace_total} (attempted: {len(done_ids)})", flush=True)

    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return traces


# ==============================================================================
# 4. LoRA SFT Training with Custom ReadoutTrainer
# ==============================================================================

class MixDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[dict], tokenizer: Any, max_seq_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = self.rows[i]
        ex = r["ex"]
        prompt = prompt_for(ex)
        if r.get("trace"):
            text = prompt + "\nReasoning: " + r["trace"]["reasoning"] + "\nFinal answer:"
        else:
            text = prompt

        tok = self.tokenizer(text, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        ids = tok["input_ids"][0]
        labels = ids.clone()

        if r.get("trace"):
            pids = self.tokenizer(prompt + "\nReasoning:", add_special_tokens=False)["input_ids"]
            labels[: min(len(pids), len(labels))] = -100
        else:
            labels[:] = -100

        return {
            "input_ids": ids,
            "attention_mask": tok["attention_mask"][0],
            "labels": labels,
            "readout_pos": len(ids) - 1,
            "n_options": len(ex["options"]),
            "answer": ex["label"],
        }


class DataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        mx = max(len(z["input_ids"]) for z in items)
        ids, masks, labs = [], [], []
        for z in items:
            n = mx - len(z["input_ids"])
            ids.append(torch.cat([z["input_ids"], torch.full((n,), self.pad_token_id, dtype=torch.long)]))
            masks.append(torch.cat([z["attention_mask"], torch.zeros(n, dtype=torch.long)]))
            labs.append(torch.cat([z["labels"], torch.full((n,), -100, dtype=torch.long)]))

        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labs),
            "readout_pos": torch.tensor([z["readout_pos"] for z in items]),
            "n_options": torch.tensor([z["n_options"] for z in items]),
            "answer": torch.tensor([z["answer"] for z in items]),
        }


class ReadoutTrainer(Trainer):
    def __init__(self, letter_ids: Dict[str, int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.letter_ids = letter_ids

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        pos = inputs.pop("readout_pos")
        nopts = inputs.pop("n_options")
        answer = inputs.pop("answer")

        out = model(**inputs, labels=labels)
        opt_ids = torch.tensor(
            [self.letter_ids[x] for x in LETTERS[: int(nopts.max())]],
            device=out.logits.device,
        )

        batch_idx = torch.arange(len(pos), device=out.logits.device)
        z = out.logits[batch_idx, pos][:, opt_ids].float()
        mask = torch.arange(z.shape[1], device=z.device)[None, :] >= nopts.to(z.device)[:, None]
        z = z.masked_fill(mask, -1e4)

        readout_loss = torch.nn.functional.cross_entropy(z, answer.to(z.device))
        lm_loss = out.loss if out.loss is not None else 0.0
        total_loss = lm_loss + readout_loss

        return (total_loss, out) if return_outputs else total_loss


def train_lora_adapter(
    args: argparse.Namespace, data: Dict[str, Dict[str, List[dict]]], traces: List[dict]
) -> Path:
    adapter_dir = args.root / "checkpoints" / args.adapter_name
    if (adapter_dir / "adapter_config.json").exists() and not args.force_retrain:
        print(f"[Training] Adapter already exists at {adapter_dir}. Skipping training.", flush=True)
        return adapter_dir

    print(f"\n[Training] Initializing LoRA fine-tuning for {args.base_model}...", flush=True)
    base_model, tokenizer, letter_ids, device = load_model_and_tokenizer(args.base_model)

    # Discover linear target modules for LoRA
    linear_names = set()
    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if any(k in leaf.lower() for k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                linear_names.add(leaf)

    print(f"  Target LoRA modules: {sorted(linear_names)}", flush=True)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=sorted(linear_names),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    train_examples = [x for t in TASKS[:-1] for x in data[t]["train"]]
    train_ids = {x["id"] for x in train_examples}
    rows = [{"ex": x} for x in train_examples] + [
        {"ex": next(y for y in train_examples if y["id"] == t["id"]), "trace": t}
        for t in traces if t["id"] in train_ids
    ]
    random.Random(args.seed).shuffle(rows)

    epochs = 1 if args.smoke_test else args.epochs
    grad_accum = 2 if args.smoke_test else args.grad_accum

    train_dataset = MixDataset(rows, tokenizer, args.max_seq_len)
    training_args = TrainingArguments(
        output_dir=str(args.root / "checkpoints" / "trainer_tmp"),
        num_train_epochs=epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        fp16=(device == "cuda" and not torch.cuda.is_bf16_supported()),
        bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    trainer = ReadoutTrainer(
        letter_ids=letter_ids,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(tokenizer.pad_token_id),
    )

    trainer.train()

    adapter_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[Training] Saved trained LoRA adapter to {adapter_dir}\n", flush=True)

    del model, base_model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adapter_dir


# ==============================================================================
# 5. Profiling, Metrics, and Online Gated Inference
# ==============================================================================

class ForwardCounter:
    def __init__(self) -> None:
        self.count = 0


def count_forward_hook(model: torch.nn.Module, counter: ForwardCounter):
    orig = model.forward

    def hooked(*args, **kwargs):
        counter.count += 1
        return orig(*args, **kwargs)

    model.forward = hooked
    return orig


def restricted_probabilities(logits: torch.Tensor, n_options: int, letter_ids: Dict[str, int]) -> np.ndarray:
    ids = torch.tensor([letter_ids[x] for x in LETTERS[:n_options]], device=logits.device)
    return torch.softmax(logits[ids].float(), dim=-1).detach().cpu().numpy()


def run_inference_pass(
    model, tokenizer, letter_ids, device: str, ex: Dict[str, Any], mode: str, max_reason_tokens: int
) -> Dict[str, Any]:
    counter = ForwardCounter()
    orig_forward = count_forward_hook(model, counter)
    try:
        sync_cuda()
        t0 = time.perf_counter()

        # Fast prompt
        base_prompt = prompt_for(ex, reasoning=False)
        inp = tokenizer(base_prompt, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True, use_cache=True)
        fast_probs = restricted_probabilities(out.logits[0, -1], len(ex["options"]), letter_ids)

        if mode == "fast":
            sync_cuda()
            return {
                "probs": fast_probs.tolist(),
                "generated_tokens": 0,
                "forward_calls": counter.count,
                "seconds": time.perf_counter() - t0,
                "escalated": False,
            }

        # Reasoning prompt
        r_prompt = prompt_for(ex, reasoning=True)
        r_inp = tokenizer(r_prompt, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            gen = model.generate(
                **r_inp,
                max_new_tokens=max_reason_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        n_gen = int(gen.shape[1] - r_inp["input_ids"].shape[1])
        scratch = tokenizer.decode(gen[0, r_inp["input_ids"].shape[1] :], skip_special_tokens=True)

        full = r_prompt + scratch + "\nFinal answer:"
        final_inp = tokenizer(full, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            final_out = model(**final_inp, output_hidden_states=False)

        probs = restricted_probabilities(final_out.logits[0, -1], len(ex["options"]), letter_ids)
        sync_cuda()
        return {
            "probs": probs.tolist(),
            "generated_tokens": n_gen,
            "forward_calls": counter.count,
            "seconds": time.perf_counter() - t0,
            "escalated": True,
        }
    finally:
        model.forward = orig_forward


def run_reasoning_tail(
    model, tokenizer, letter_ids, device: str, ex: Dict[str, Any], max_reason_tokens: int
) -> Dict[str, Any]:
    counter = ForwardCounter()
    orig = count_forward_hook(model, counter)
    try:
        sync_cuda()
        t0 = time.perf_counter()
        r_prompt = prompt_for(ex, reasoning=True)
        r_inp = tokenizer(r_prompt, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            gen = model.generate(
                **r_inp,
                max_new_tokens=max_reason_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        n_gen = int(gen.shape[1] - r_inp["input_ids"].shape[1])
        scratch = tokenizer.decode(gen[0, r_inp["input_ids"].shape[1] :], skip_special_tokens=True)

        final_inp = tokenizer(r_prompt + scratch + "\nFinal answer:", return_tensors="pt").to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            final_out = model(**final_inp, output_hidden_states=False)

        probs = restricted_probabilities(final_out.logits[0, -1], len(ex["options"]), letter_ids)
        sync_cuda()
        return {
            "probs": probs.tolist(),
            "generated_tokens": n_gen,
            "forward_calls": counter.count,
            "seconds": time.perf_counter() - t0,
            "escalated": True,
        }
    finally:
        model.forward = orig


def compute_metrics(rows: List[Dict[str, Any]], bins: int = 10) -> Dict[str, float]:
    acc = float(np.mean([int(np.argmax(r["probs"]) == r["label"]) for r in rows]))
    nll = float(np.mean([-math.log(max(r["probs"][r["label"]], 1e-12)) for r in rows]))

    brier_vals = []
    for r in rows:
        target = np.zeros(len(r["probs"]))
        target[r["label"]] = 1.0
        brier_vals.append(np.sum((np.asarray(r["probs"]) - target) ** 2))
    brier = float(np.mean(brier_vals))

    # Expected Calibration Error (ECE)
    conf = np.asarray([max(r["probs"]) for r in rows])
    ok = np.asarray([int(np.argmax(r["probs"]) == r["label"]) for r in rows])
    ece = 0.0
    for lo in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        hi = lo + 1.0 / bins
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(ok[mask].mean()) - float(conf[mask].mean()))

    return {
        "n": len(rows),
        "accuracy": acc,
        "ece": ece,
        "nll": nll,
        "brier": brier,
        "avg_tokens": float(np.mean([r["generated_tokens"] for r in rows])),
        "avg_seconds": float(np.mean([r["seconds"] for r in rows])),
        "avg_forward_calls": float(np.mean([r["forward_calls"] for r in rows])),
    }


def select_gate_threshold(
    val_fast: List[Dict[str, Any]], val_reason: List[Dict[str, Any]], gap: float
) -> Dict[str, float]:
    reason_acc = float(np.mean([int(np.argmax(r["probs"]) == r["label"]) for r in val_reason]))
    candidates = []

    for th in np.linspace(0.0, 1.0, 101):
        gated = []
        escalated = 0
        for f, r in zip(val_fast, val_reason):
            use_r = max(f["probs"]) < th
            gated.append(r if use_r else f)
            escalated += int(use_r)
        acc = float(np.mean([int(np.argmax(x["probs"]) == x["label"]) for x in gated]))
        rate = escalated / len(gated)
        candidates.append((rate, -acc, th, acc))

    eligible = [x for x in candidates if x[3] >= reason_acc - gap]
    chosen = min(eligible or candidates, key=lambda x: (x[0], x[1]))
    return {
        "threshold": float(chosen[2]),
        "val_reasoning_acc": reason_acc,
        "val_gated_acc": float(chosen[3]),
        "val_escalation_rate": float(chosen[0]),
    }


def evaluate_pipeline(
    args: argparse.Namespace, data: Dict[str, Dict[str, List[dict]]], adapter_dir: Path
) -> None:
    print(f"\n[Evaluation] Loading model with adapter {adapter_dir}...", flush=True)
    model, tokenizer, letter_ids, device = load_model_and_tokenizer(args.base_model, adapter_dir)

    val_examples = [x for t in TASKS for x in data[t]["val"]]
    test_examples = [x for t in TASKS for x in data[t]["test"]]
    print(f"  Validation examples: {len(val_examples)} | Test examples: {len(test_examples)}", flush=True)

    eval_cache_dir = args.root / "cache" / "eval"
    eval_cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache_file = eval_cache_dir / "val_cache.json"

    # 1. Validation evaluation for Gate Calibration
    if val_cache_file.exists():
        print(f"[Evaluation] Found cached validation results at {val_cache_file}. Loading...", flush=True)
        cached_val = json.loads(val_cache_file.read_text(encoding="utf-8"))
        val_fast = cached_val["val_fast"]
        val_reason = cached_val["val_reason"]
        gate = cached_val["gate"]
    else:
        print("\n[Evaluation] Running validation passes to calibrate confidence gate threshold...", flush=True)
        val_fast, val_reason = [], []
        for i, ex in enumerate(val_examples):
            vf = run_inference_pass(model, tokenizer, letter_ids, device, ex, "fast", args.max_reason_tokens)
            vr = run_inference_pass(model, tokenizer, letter_ids, device, ex, "reasoning", args.max_reason_tokens)
            val_fast.append({"id": ex["id"], "task": ex["task"], "label": ex["label"], **vf})
            val_reason.append({"id": ex["id"], "task": ex["task"], "label": ex["label"], **vr})
            if (i + 1) % 25 == 0 or (i + 1) == len(val_examples):
                print(f"  Validation progress: {i + 1}/{len(val_examples)}", flush=True)

        gate = select_gate_threshold(val_fast, val_reason, args.target_accuracy_gap)
        val_cache_file.write_text(
            json.dumps({"val_fast": val_fast, "val_reason": val_reason, "gate": gate}),
            encoding="utf-8"
        )

    threshold = gate["threshold"]
    print(f"[Gate] Calibrated Gate Threshold: {threshold:.4f} (Val Escalation Rate: {gate['val_escalation_rate']:.2%})", flush=True)

    # 2. Test Evaluation (Fast, Reasoning, and Online Gated)
    test_cache_file = eval_cache_dir / "test_cache.json"
    test_fast, test_reason, test_gated = [], [], []
    start_idx = 0
    if test_cache_file.exists():
        try:
            cached_test = json.loads(test_cache_file.read_text(encoding="utf-8"))
            test_fast = cached_test.get("test_fast", [])
            test_reason = cached_test.get("test_reason", [])
            test_gated = cached_test.get("test_gated", [])
            start_idx = len(test_gated)
            print(f"[Evaluation] Resuming test evaluation from cached progress: {start_idx}/{len(test_examples)}", flush=True)
        except Exception as e:
            print(f"[Evaluation] Warning: Could not read test cache ({e}). Starting fresh.", flush=True)
            test_fast, test_reason, test_gated = [], [], []
            start_idx = 0

    if start_idx < len(test_examples):
        print(f"\n[Evaluation] Executing test evaluation across fast, reasoning, and gated modes ({start_idx}/{len(test_examples)} completed)...", flush=True)
        for i in range(start_idx, len(test_examples)):
            ex = test_examples[i]
            # Fast mode
            tf = run_inference_pass(model, tokenizer, letter_ids, device, ex, "fast", args.max_reason_tokens)
            test_fast.append({"id": ex["id"], "task": ex["task"], "label": ex["label"], **tf})

            # Reasoning mode
            tr = run_inference_pass(model, tokenizer, letter_ids, device, ex, "reasoning", args.max_reason_tokens)
            test_reason.append({"id": ex["id"], "task": ex["task"], "label": ex["label"], **tr})

            # Online Gated mode (fast pass -> escalate to reasoning tail if confidence < threshold)
            if max(tf["probs"]) < threshold:
                tail = run_reasoning_tail(model, tokenizer, letter_ids, device, ex, args.max_reason_tokens)
                gated_res = {
                    "probs": tail["probs"],
                    "generated_tokens": tail["generated_tokens"],
                    "forward_calls": tf["forward_calls"] + tail["forward_calls"],
                    "seconds": tf["seconds"] + tail["seconds"],
                    "escalated": True,
                }
            else:
                gated_res = {
                    "probs": tf["probs"],
                    "generated_tokens": 0,
                    "forward_calls": tf["forward_calls"],
                    "seconds": tf["seconds"],
                    "escalated": False,
                }

            test_gated.append({"id": ex["id"], "task": ex["task"], "label": ex["label"], **gated_res})

            if (i + 1) % 25 == 0 or (i + 1) == len(test_examples):
                print(f"  Test progress: {i + 1}/{len(test_examples)}", flush=True)
                test_cache_file.write_text(json.dumps({
                    "test_fast": test_fast,
                    "test_reason": test_reason,
                    "test_gated": test_gated,
                }), encoding="utf-8")

    # Compute summaries
    metrics_fast = compute_metrics(test_fast[args.warmup :])
    metrics_reason = compute_metrics(test_reason[args.warmup :])
    metrics_gated = compute_metrics(test_gated[args.warmup :])
    metrics_gated["escalation_rate"] = float(np.mean([r["escalated"] for r in test_gated[args.warmup :]]))
    metrics_gated["reasoning_call_reduction"] = 1.0 - metrics_gated["escalation_rate"]

    # Per-task breakdown table
    table_rows = []
    for task in TASKS + ["overall"]:
        tf_sub = [r for r in test_fast[args.warmup :] if task == "overall" or r["task"] == task]
        tr_sub = [r for r in test_reason[args.warmup :] if task == "overall" or r["task"] == task]
        tg_sub = [r for r in test_gated[args.warmup :] if task == "overall" or r["task"] == task]

        mf = compute_metrics(tf_sub)
        mr = compute_metrics(tr_sub)
        mg = compute_metrics(tg_sub)

        table_rows.append({
            "task": task,
            "n": len(tf_sub),
            "fast_acc": mf["accuracy"],
            "fast_ece": mf["ece"],
            "fast_brier": mf["brier"],
            "reason_acc": mr["accuracy"],
            "reason_ece": mr["ece"],
            "reason_brier": mr["brier"],
            "gated_acc": mg["accuracy"],
            "escalation_rate": float(np.mean([r["escalated"] for r in tg_sub])),
            "avg_tokens": mg["avg_tokens"],
            "avg_seconds": mg["avg_seconds"],
        })

    df = pd.DataFrame(table_rows)
    metrics_dir = args.root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df_path = metrics_dir / "final_table.csv"
    df.to_csv(df_path, index=False)

    results_payload = {
        "metadata": {
            "base_model": args.base_model,
            "adapter": str(adapter_dir),
            "device": device,
            "smoke_test": args.smoke_test,
            "gate_threshold": threshold,
        },
        "gate_calibration": gate,
        "overall_test": {
            "fast": metrics_fast,
            "reasoning": metrics_reason,
            "gated": metrics_gated,
        },
        "per_task_table": table_rows,
    }

    json_path = metrics_dir / "efficiency.json"
    json_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("FINAL RESULTS & EFFICIENCY TRADE-OFF:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"[Results] CSV table saved to: {df_path}")
    print(f"[Results] JSON metrics saved to: {json_path}\n")


# ==============================================================================
# Main Orchestration
# ==============================================================================

def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    cache_dir = args.root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (cache_dir / "hub").mkdir(parents=True, exist_ok=True)
    (args.root / "tmp").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "hub")
    os.environ["TMPDIR"] = str(args.root / "tmp")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 80)
    print("JEV ESCALATION HPC PIPELINE")
    print(f"Working Directory: {args.root}")
    print(f"Base Model:        {args.base_model}")
    print(f"Smoke Test:        {args.smoke_test}")
    print(f"Device:            {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80 + "\n", flush=True)

    # 1. Prepare data splits
    data = prepare_data_splits(args)

    adapter_path = args.root / "checkpoints" / args.adapter_name

    # 2. Training (if requested or neither flag specified)
    if not args.eval_only:
        traces = generate_teacher_traces(args, data)
        adapter_path = train_lora_adapter(args, data, traces)

    # 3. Evaluation & Profiling (if requested or neither flag specified)
    if not args.train_only:
        evaluate_pipeline(args, data, adapter_path)

    print("[Pipeline] Finished successfully!", flush=True)


if __name__ == "__main__":
    main()
