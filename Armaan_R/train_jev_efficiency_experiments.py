#!/usr/bin/env python3
"""Compare Qwen fine-tuning/compression strategies and Jev-style escalation.

This is a standalone companion to train_jev_qwin_thingy.py. It trains/evaluates
several model variants against shared data splits and writes versioned results
under JEV_OUTPUT_DIR (default: ./jev_efficiency_experiments).

Examples:
  python train_jev_efficiency_experiments.py --smoke
  JEV_SMOKE_TEST=0 python train_jev_efficiency_experiments.py
  python train_jev_efficiency_experiments.py --variants lora_all,pruned25_lora

The runtime requires torch, transformers, datasets, peft, scikit-learn, numpy,
pandas, matplotlib, and scipy in the active Python environment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import time
from pathlib import Path
from typing import Any

SEED = 2026
PRIMARY_MODEL = "Qwen/Qwen3-1.7B"
SMALL_MODEL = "Qwen/Qwen3-0.6B"
TEACHER_MODEL = "Qwen/Qwen3-4B"
TASKS = ["sentiment", "inference", "science questions", "general knowledge", "word math"]
SEEN_TASKS = TASKS[:-1]
LETTERS = list("ABCDE")
SCRIPT_VERSION = "1.6"
GATE_ESCALATION_COST = 0.02  # utility = accuracy - cost * escalation_rate

DESCRIPTIONS = {
    "sentiment": "Classify the sentiment of the review as positive or negative.",
    "inference": "Determine whether the hypothesis is entailed, contradicted, or neutral given the premise.",
    "science questions": "Answer the grade school science question.",
    "general knowledge": "Answer the multiple-choice exam question.",
    "word math": "Solve the math word problem.",
}

VARIANTS: dict[str, dict[str, Any]] = {
    "frozen_1_7b": {"model": PRIMARY_MODEL, "training": "none"},
    "frozen_0_6b": {"model": SMALL_MODEL, "training": "none"},
    "lora_all": {"model": PRIMARY_MODEL, "training": "lora_all"},
    "lora_attention": {"model": PRIMARY_MODEL, "training": "lora_attention"},
    "lora_mlp": {"model": PRIMARY_MODEL, "training": "lora_mlp"},
    "top25_finetune": {"model": PRIMARY_MODEL, "training": "top25"},
    "pruned25_lora": {"model": PRIMARY_MODEL, "training": "pruned25_lora", "prune_fraction": 0.25},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_smoke = os.environ.get("JEV_SMOKE_TEST", "1").lower() not in {"0", "false", "no"}
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", dest="smoke", action="store_true", help="Use small samples and one epoch (default).")
    group.add_argument("--full", dest="smoke", action="store_false", help="Use the full experiment sample sizes.")
    parser.set_defaults(smoke=default_smoke)
    parser.add_argument("--output-dir", default=os.environ.get("JEV_OUTPUT_DIR", "./jev_efficiency_experiments"))
    parser.add_argument("--variants", default=",".join(VARIANTS), help="Comma-separated variant ids; default runs all.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-reason-tokens", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate for trained variants (default: 5e-5).")
    parser.add_argument("--max-trace-examples", type=int, default=None)
    parser.add_argument("--force-retrain", action="store_true", help="Ignore trained artifacts and train selected variants again.")
    return parser.parse_args()


def configure_hf_cache() -> str | None:
    """Use one scratch cache across Slurm jobs so Hub downloads can resume."""
    job_id = os.environ.get("SLURM_JOB_ID")
    user = os.environ.get("USER")
    scratch = Path("/scratch") / user if user else None
    current = os.environ.get("HF_HOME")
    persistent = os.environ.get("JEV_HF_CACHE_DIR")
    if not persistent and scratch and scratch.is_dir():
        persistent = str(scratch / "huggingface-cache")
    if job_id and persistent and current and job_id in current:
        print(f"HF_HOME={current} is job-specific; using persistent cache {persistent} instead.")
        os.environ["HF_HOME"] = persistent
    elif not current and job_id and persistent:
        os.environ["HF_HOME"] = persistent
    selected = os.environ.get("HF_HOME")
    if selected:
        print(f"Hugging Face cache root: {selected}")
    return selected


def load_runtime() -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    return locals()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def file_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def setup(args: argparse.Namespace, rt: dict[str, Any]) -> dict[str, Any]:
    np, torch = rt["np"], rt["torch"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rt["set_seed"](args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = bool(device == "cuda" and torch.cuda.is_bf16_supported())
    precision = "bf16" if use_bf16 else ("fp16" if device == "cuda" else "fp32")
    config = {
        "script_version": SCRIPT_VERSION,
        "seed": args.seed,
        "smoke": args.smoke,
        "models": {"primary": PRIMARY_MODEL, "small": SMALL_MODEL, "teacher": TEACHER_MODEL},
        "max_reason_tokens": args.max_reason_tokens,
        "max_seq_len": 1024,
        "epochs": 1 if args.smoke else 2,
        "training_precision": precision,
        "learning_rate": args.learning_rate,
        "train_per_task": 20 if args.smoke else 2000,
        "val_per_task": 20 if args.smoke else 300,
        "test_per_task": 20 if args.smoke else 500,
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.05},
        "gate_escalation_cost": GATE_ESCALATION_COST,
        "hf_hub_disable_xet": os.environ.get("HF_HUB_DISABLE_XET", "0"),
        "hf_hub_download_timeout": os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "default"),
    }
    config["teacher_model_requested"] = (os.environ.get("JEV_TEACHER_MODEL") or
                                          (PRIMARY_MODEL if args.smoke else TEACHER_MODEL))
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID")
    hf_home = os.environ.get("HF_HOME", "")
    if job_id and hf_home and job_id in hf_home:
        print("WARNING: HF_HOME contains this Slurm job id. Partial model downloads will not be reused by a later job; "
              "set HF_HOME to a persistent path such as /scratch/$USER/huggingface-cache.")
    for name in ("data", "traces", "variants", "metrics", "plots", "reports"):
        (out / name).mkdir(exist_ok=True)
    print(f"Output directory: {out}\nDevice: {device}; smoke: {args.smoke}")
    model_dtype = (torch.bfloat16 if precision == "bf16" else
                   torch.float16 if precision == "fp16" else torch.float32)
    return {"root": out, "device": device, "dtype": model_dtype, "precision": precision, "config": config}


def norm_text(value: Any) -> str:
    return str(value).strip()


def letters_to_label(answer: Any, labels: list[str]) -> int | None:
    value = str(answer).strip().upper().replace("(", "").replace(")", "").replace(".", "")
    if value in labels:
        return labels.index(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def map_row(task: str, row: dict[str, Any]) -> tuple[str, list[str], int] | None:
    if task == "sentiment":
        return "Review: " + norm_text(row["sentence"]), ["negative", "positive"], int(row["label"])
    if task == "inference":
        return ("Premise: " + norm_text(row["premise"]) + "\nHypothesis: " + norm_text(row["hypothesis"]),
                ["entailment", "neutral", "contradiction"], int(row["label"]))
    if task == "science questions":
        choices = row["choices"]
        labels = list(map(str, choices["label"]))
        answer = str(row["answerKey"])
        return norm_text(row["question"]), list(map(norm_text, choices["text"])), labels.index(answer)
    if task == "general knowledge":
        return norm_text(row["question"]), list(map(norm_text, row["choices"])), int(row["answer"])
    if task == "word math":
        raw_options = list(map(norm_text, row["options"]))
        labels = [x.split(")", 1)[0].strip(" (") for x in raw_options]
        options = [x.split(")", 1)[1].strip() if ")" in x else x for x in raw_options]
        label = letters_to_label(row["correct"], labels)
        if label is None:
            return None
        return norm_text(row["question"]), options, label
    raise ValueError(task)


def make_record(task: str, row: dict[str, Any], seed: int) -> dict[str, Any] | None:
    mapped = map_row(task, row)
    if mapped is None:
        return None
    question, options, label = mapped
    if not options or label < 0 or label >= len(options) or len(options) > len(LETTERS):
        return None
    shuffle_seed = int(hashlib.sha256(f"{seed}:{task}:{question}".encode()).hexdigest()[:8], 16)
    order = list(range(len(options)))
    random.Random(shuffle_seed).shuffle(order)
    shuffled_options = [options[i] for i in order]
    shuffled_label = order.index(label)
    payload = f"{task}\n{question}\n" + "\n".join(shuffled_options) + f"\n{shuffled_label}"
    return {"id": hashlib.sha256(payload.encode()).hexdigest(), "task": task, "question": question,
            "options": shuffled_options, "label": shuffled_label}


def sample_split(split: Any, count: int, seed: int) -> list[dict[str, Any]]:
    indices = list(range(len(split)))
    random.Random(seed).shuffle(indices)
    return [split[i] for i in indices[:count]]


def load_data(ctx: dict[str, Any], rt: dict[str, Any], args: argparse.Namespace) -> dict[str, dict[str, list[dict[str, Any]]]]:
    load_dataset = rt["load_dataset"]
    count = ctx["config"]["train_per_task"]
    val_count = ctx["config"]["val_per_task"]
    test_count = ctx["config"]["test_per_task"]
    sources = {
        "sentiment": ("stanfordnlp/sst2", None, "train", "validation", "validation"),
        "inference": ("facebook/anli", None, "train_r1", "dev_r1", "test_r1"),
        "science questions": ("allenai/ai2_arc", "ARC-Challenge", "train", "validation", "test"),
        "general knowledge": ("cais/mmlu", "all", "auxiliary_train", "dev", "test"),
        "word math": ("deepmind/aqua_rat", "raw", "train", "validation", "test"),
    }
    data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task, (dataset, subset, train_key, val_key, test_key) in sources.items():
        raw = load_dataset(dataset, subset) if subset else load_dataset(dataset)
        if task == "sentiment":
            pool = raw["validation"]
            ids = list(range(len(pool)))
            random.Random(args.seed).shuffle(ids)
            val_raw = [pool[i] for i in ids[:min(val_count, len(ids) // 2)]]
            test_raw = [pool[i] for i in ids[min(val_count, len(ids) // 2):min(val_count, len(ids) // 2) + test_count]]
            train_raw = sample_split(raw["train"], count, args.seed)
        else:
            train_raw = sample_split(raw[train_key], count, args.seed)
            val_raw = sample_split(raw[val_key], val_count, args.seed + 1)
            test_raw = sample_split(raw[test_key], test_count, args.seed + 2)
        map_rows = lambda rows: [r for r in (make_record(task, x, args.seed) for x in rows) if r]
        data[task] = {"train": map_rows(train_raw), "val": map_rows(val_raw), "test": map_rows(test_raw)}
        print(task, {key: len(value) for key, value in data[task].items()})

    train_ids = {r["id"] for task in SEEN_TASKS for r in data[task]["train"]}
    val_ids = {r["id"] for task in TASKS for r in data[task]["val"]}
    test_ids = {r["id"] for task in TASKS for r in data[task]["test"]}
    if train_ids & (val_ids | test_ids):
        raise RuntimeError("Data split leakage detected")
    if any(r["task"] == "word math" for t in SEEN_TASKS for r in data[t]["train"]):
        raise RuntimeError("Held-out task entered training")
    write_json(ctx["root"] / "data" / f"splits_{digest(ctx['config'])[:12]}.json", data)
    ctx["data"] = data
    return data


def get_tokenizer(model_name: str, rt: dict[str, Any]) -> Any:
    tokenizer = rt["AutoTokenizer"].from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_letter_ids(tokenizer: Any) -> dict[str, int]:
    letter_ids: dict[str, int] = {}
    for letter in LETTERS:
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(ids) != 1:
            ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Answer choice {letter} is not a single token: {ids}")
        letter_ids[letter] = int(ids[0])
    return letter_ids


def discover_layers(model: Any) -> tuple[Any, str]:
    for attr_path in ("model.layers", "model.model.layers", "transformer.h"):
        node = model
        try:
            for part in attr_path.split("."):
                node = getattr(node, part)
            return node, attr_path
        except AttributeError:
            continue
    raise RuntimeError("Could not find transformer block list for layer pruning/freezing")


def evenly_prune_layers(model: Any, fraction: float) -> dict[str, int]:
    layers, path = discover_layers(model)
    original = len(layers)
    keep = max(1, round(original * (1.0 - fraction)))
    selected = sorted(set(round(i * original / keep) for i in range(keep)))[:keep]
    new_layers = [layers[i] for i in selected]
    parent = model
    parts = path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], __import__("torch").nn.ModuleList(new_layers))
    model.config.num_hidden_layers = len(new_layers)
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.num_hidden_layers = len(new_layers)
    return {"original_layers": original, "remaining_layers": len(new_layers), "removed_layers": original - len(new_layers)}


def model_load(model_name: str, ctx: dict[str, Any], rt: dict[str, Any]) -> Any:
    torch = rt["torch"]
    kwargs = {"torch_dtype": ctx["dtype"], "trust_remote_code": True}
    if ctx["device"] == "cuda":
        kwargs["device_map"] = "auto"
    else:
        kwargs["low_cpu_mem_usage"] = True
    model = rt["AutoModelForCausalLM"].from_pretrained(model_name, **kwargs)
    return model


def prompt_for(ex: dict[str, Any], reasoning: bool = False) -> str:
    body = f"Task: {DESCRIPTIONS[ex['task']]}\nQuestion: {ex['question']}\nOptions:\n"
    body += "".join(f"{LETTERS[i]}. {option}\n" for i, option in enumerate(ex["options"]))
    body += "Answer:"
    return body + ("\nReasoning:" if reasoning else "")


def load_or_make_traces(ctx: dict[str, Any], rt: dict[str, Any], args: argparse.Namespace,
                        tokenizer: Any, data: dict[str, Any]) -> list[dict[str, Any]]:
    np = rt["np"]
    max_traces = args.max_trace_examples or (40 if args.smoke else 1500)
    eligible = [r for task in SEEN_TASKS for r in data[task]["train"]]
    random.Random(args.seed).shuffle(eligible)
    eligible = eligible[:max_traces]
    requested_teacher = os.environ.get("JEV_TEACHER_MODEL") or (PRIMARY_MODEL if args.smoke else TEACHER_MODEL)
    teacher_candidates = [requested_teacher]
    if requested_teacher != PRIMARY_MODEL:
        teacher_candidates.append(PRIMARY_MODEL)
    fingerprint = digest({"ids": [r["id"] for r in eligible], "teacher_candidates": teacher_candidates,
                          "seed": args.seed})
    path = ctx["root"] / "traces" / f"teacher_{fingerprint[:12]}.json"
    metadata_path = path.with_suffix(".meta.json")
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ctx["teacher_model_used"] = metadata["teacher_model_used"]
        ctx["config"]["teacher_model_used"] = metadata["teacher_model_used"]
        return json.loads(path.read_text(encoding="utf-8"))
    # A trace JSON without its completion metadata is an interrupted partial run.
    path.unlink(missing_ok=True)
    teacher = None
    teacher_name = None
    load_errors = []
    for candidate in teacher_candidates:
        try:
            teacher = model_load(candidate, ctx, rt)
            teacher_name = candidate
            break
        except Exception as exc:
            load_errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            print(f"Could not load trace teacher {candidate}; trying fallback if configured.")
    if teacher is None:
        raise RuntimeError("Could not load any trace teacher. " + " | ".join(load_errors) +
                           ". The cache contains incomplete weight shards. Keep HF_HOME fixed across Slurm jobs; "
                           "the supplied Slurm launcher disables Xet to retry over standard HTTPS.")
    ctx["teacher_model_used"] = teacher_name
    ctx["config"]["teacher_model_used"] = teacher_name
    print(f"Generating training traces with {teacher_name}")
    accepted: list[dict[str, Any]] = []
    torch = rt["torch"]
    for index, ex in enumerate(eligible, 1):
        ask = prompt_for(ex) + "\nReason in under 150 words. End with exactly: Final answer: X"
        enc = tokenizer(ask, return_tensors="pt").to(teacher.device)
        with torch.no_grad():
            gen = teacher.generate(**enc, max_new_tokens=220, do_sample=False, use_cache=True,
                                   pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        import re
        matches = re.findall(r"Final answer:\s*([ABCDE])", completion, re.I)
        if matches and LETTERS.index(matches[-1].upper()) == ex["label"]:
            cutoff = completion.lower().rfind("final answer:")
            accepted.append({"id": ex["id"], "task": ex["task"], "reasoning": completion[:cutoff].strip(),
                             "answer": LETTERS[ex["label"]]})
        if index % 5 == 0 or index == len(eligible):
            write_json(path, accepted)
            print(f"Teacher traces {index}/{len(eligible)}; accepted {len(accepted)}")
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_json(metadata_path, {"teacher_model_used": teacher_name,
                               "eligible_examples": len(eligible), "accepted_traces": len(accepted),
                               "complete": True})
    return accepted


def model_fingerprint(variant: str, spec: dict[str, Any], ctx: dict[str, Any], args: argparse.Namespace,
                      trace_signature: list[str]) -> str:
    return digest({"variant": variant, "spec": spec, "config": ctx["config"],
                   "trace_signature": sorted(trace_signature), "seed": args.seed})


def train_variant(variant: str, spec: dict[str, Any], fingerprint: str, traces: list[dict[str, Any]],
                  data: dict[str, Any], tokenizer: Any, ctx: dict[str, Any], rt: dict[str, Any],
                  args: argparse.Namespace) -> tuple[Path | None, dict[str, Any]]:
    torch = rt["torch"]
    artifact_dir = ctx["root"] / "variants" / variant / fingerprint
    adapter_dir = artifact_dir / "adapter"
    model_dir = artifact_dir / "pruned_base"
    manifest = artifact_dir / "manifest.json"
    if spec["training"] == "none":
        artifact_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest, {"fingerprint": fingerprint, "variant": variant, "training": "none"})
        return None, {"training_seconds": 0.0, "trainable_parameters": 0, "pruning": None}
    if not args.force_retrain:
        if spec["training"] == "top25":
            full_model_saved = (adapter_dir / "config.json").exists() and any(
                adapter_dir.glob("model*.safetensors"))
            if full_model_saved:
                return adapter_dir, json.loads(manifest.read_text()) if manifest.exists() else {
                    "training_seconds": None, "trainable_parameters": None, "pruning": None}
        elif (adapter_dir / "adapter_config.json").exists():
            return adapter_dir, json.loads(manifest.read_text()) if manifest.exists() else {
                "training_seconds": None, "trainable_parameters": None, "pruning": None}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model_name = spec["model"]
    if spec["training"] == "pruned25_lora" and not args.force_retrain and (model_dir / "config.json").exists():
        model = rt["AutoModelForCausalLM"].from_pretrained(model_dir, torch_dtype=ctx["dtype"],
                                                            device_map="auto" if ctx["device"] == "cuda" else None,
                                                            trust_remote_code=True)
        pruning = json.loads(manifest.read_text()).get("pruning") if manifest.exists() else None
    else:
        model = model_load(model_name, ctx, rt)
        pruning = evenly_prune_layers(model, float(spec["prune_fraction"])) if spec["training"] == "pruned25_lora" else None
        if pruning:
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

    if spec["training"] in {"lora_all", "lora_attention", "lora_mlp", "pruned25_lora"}:
        names: set[str] = set()
        wanted = ("q_proj", "k_proj", "v_proj", "o_proj") if spec["training"] == "lora_attention" else (
            ("gate_proj", "up_proj", "down_proj") if spec["training"] == "lora_mlp" else
            ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in wanted:
                names.add(name.split(".")[-1])
        if not names:
            raise RuntimeError(f"No projection targets discovered for {variant}")
        cfg = rt["LoraConfig"](r=16, lora_alpha=32, lora_dropout=0.05, target_modules=sorted(names), task_type="CAUSAL_LM")
        model = rt["get_peft_model"](model, cfg)
    elif spec["training"] == "top25":
        layers, _ = discover_layers(model)
        freeze_count = max(0, int(len(layers) * 0.75))
        for parameter in model.parameters():
            parameter.requires_grad = False
        for layer in list(layers)[freeze_count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        # Keep final normalization and vocabulary head trainable with the top blocks.
        for name, parameter in model.named_parameters():
            if any(key in name.lower() for key in ("norm", "lm_head")):
                parameter.requires_grad = True
    else:
        raise ValueError(f"Unsupported training recipe: {spec['training']}")

    train_rows: list[dict[str, Any]] = [{"ex": r, "trace": None} for task in SEEN_TASKS for r in data[task]["train"]]
    by_id = {r["id"]: r for task in SEEN_TASKS for r in data[task]["train"]}
    train_ids = set(by_id)
    if any(t["id"] not in train_ids for t in traces):
        raise RuntimeError("Teacher trace refers to a non-training example")
    train_rows += [{"ex": by_id[t["id"]], "trace": t} for t in traces]
    random.Random(args.seed).shuffle(train_rows)

    class MixedDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(train_rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = train_rows[index]
            ex, trace = row["ex"], row["trace"]
            prefix = prompt_for(ex)
            text = prefix if trace is None else prefix + "\nReasoning: " + trace["reasoning"] + "\nFinal answer:"
            encoded = tokenizer(text, truncation=True, max_length=1024, return_tensors="pt")
            ids = encoded["input_ids"][0]
            labels = ids.clone()
            if trace is None:
                labels[:] = -100
            else:
                mask_text = prefix + "\nReasoning:"
                mask_len = len(tokenizer(mask_text, add_special_tokens=False)["input_ids"])
                labels[:min(mask_len, len(labels))] = -100
            return {"input_ids": ids, "attention_mask": encoded["attention_mask"][0], "labels": labels,
                    "readout_pos": len(ids) - 1, "n_options": len(ex["options"]), "answer": ex["label"]}

    class BatchCollator:
        def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
            max_len = max(len(x["input_ids"]) for x in items)
            batch: dict[str, list[Any]] = {k: [] for k in ("input_ids", "attention_mask", "labels")}
            for x in items:
                pad = max_len - len(x["input_ids"])
                batch["input_ids"].append(torch.cat([x["input_ids"], torch.full((pad,), tokenizer.pad_token_id)]))
                batch["attention_mask"].append(torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
                batch["labels"].append(torch.cat([x["labels"], torch.full((pad,), -100)]))
            result = {k: torch.stack(v) for k, v in batch.items()}
            for key in ("readout_pos", "n_options", "answer"):
                result[key] = torch.tensor([x[key] for x in items])
            return result

    train_dataset = MixedDataset()
    training_args = rt["TrainingArguments"](
        output_dir=str(artifact_dir / "trainer"), num_train_epochs=ctx["config"]["epochs"],
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1, gradient_accumulation_steps=2 if args.smoke else 8,
        fp16=(ctx["precision"] == "fp16"), bf16=(ctx["precision"] == "bf16"),
        logging_steps=10, save_strategy="no", report_to="none",
        max_grad_norm=1.0, remove_unused_columns=False, dataloader_num_workers=0)

    class ReadoutTrainer(rt["Trainer"]):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
                         num_items_in_batch: Any = None) -> Any:
            labels = inputs.pop("labels")
            positions = inputs.pop("readout_pos")
            n_options = inputs.pop("n_options")
            answers = inputs.pop("answer")
            out = model(**inputs, output_hidden_states=False)
            logits = out.logits
            valid = labels[:, 1:] != -100
            if bool(valid.any()):
                lm_loss = torch.nn.functional.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                                                             labels[:, 1:].reshape(-1), ignore_index=-100)
            else:
                # Direct-answer rows have no text-generation targets; avoid all-ignored CE (NaN).
                lm_loss = logits.sum() * 0.0
            letter_ids = torch.tensor([ctx["letter_ids"][letter] for letter in LETTERS[:int(n_options.max())]],
                                      device=logits.device)
            readout = logits[torch.arange(len(positions), device=logits.device), positions][:, letter_ids].float()
            invalid = torch.arange(readout.shape[1], device=logits.device)[None, :] >= n_options.to(logits.device)[:, None]
            readout = readout.masked_fill(invalid, -1e4)
            answer_loss = torch.nn.functional.cross_entropy(readout, answers.to(logits.device))
            loss = lm_loss + answer_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss for variant {variant}")
            return (loss, out) if return_outputs else loss

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer = ReadoutTrainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=BatchCollator())
    trainer.train()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    stats = {"training_seconds": elapsed, "trainable_parameters": trainable_count,
             "pruning": pruning, "peak_training_gpu_bytes": peak}
    write_json(manifest, {"fingerprint": fingerprint, "variant": variant, **stats})
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adapter_dir, stats


def load_variant_model(variant: str, spec: dict[str, Any], artifact: Path | None, ctx: dict[str, Any], rt: dict[str, Any]) -> Any:
    if spec["training"] == "pruned25_lora":
        model_path = artifact.parent / "pruned_base" if artifact else None
        if model_path is None or not model_path.exists():
            model = model_load(spec["model"], ctx, rt)
            evenly_prune_layers(model, spec["prune_fraction"])
        else:
            model = rt["AutoModelForCausalLM"].from_pretrained(model_path, torch_dtype=ctx["dtype"],
                                                                device_map="auto" if ctx["device"] == "cuda" else None,
                                                                trust_remote_code=True)
    elif spec["training"] == "top25" and artifact is not None:
        # Top-layer fine-tuning saves a full model checkpoint, not a PEFT adapter.
        model = rt["AutoModelForCausalLM"].from_pretrained(
            artifact, torch_dtype=ctx["dtype"],
            device_map="auto" if ctx["device"] == "cuda" else None,
            trust_remote_code=True)
    else:
        model = model_load(spec["model"], ctx, rt)
    if artifact is not None and (artifact / "adapter_config.json").exists():
        model = rt["PeftModel"].from_pretrained(model, artifact)
    model.eval()
    return model


def restricted_probs(logits: Any, n_options: int, letter_ids: dict[str, int], np: Any) -> list[float]:
    torch = __import__("torch")
    ids = torch.tensor([letter_ids[x] for x in LETTERS[:n_options]], device=logits.device)
    return torch.softmax(logits[ids].float(), dim=-1).detach().cpu().numpy().astype(float).tolist()


def infer_fast(model: Any, tokenizer: Any, ex: dict[str, Any], ctx: dict[str, Any],
               rt: dict[str, Any], include_hidden: bool = True) -> dict[str, Any]:
    encoded = tokenizer(prompt_for(ex), return_tensors="pt").to(model.device)
    with rt["torch"].inference_mode():
        output = model(**encoded, output_hidden_states=include_hidden, use_cache=True)
    result = {"generated_tokens": 0}
    if include_hidden:
        hidden = output.hidden_states[-1][0, -1].float().detach().cpu().numpy()
        chunks = rt["np"].array_split(hidden, 16)
        result["hidden"] = [float(v) for chunk in chunks for v in (chunk.mean(), chunk.std())]
        result["hidden"].extend((float(hidden.mean()), float(hidden.std()), float(rt["np"].linalg.norm(hidden))))
    result["probs"] = restricted_probs(output.logits[0, -1], len(ex["options"]), ctx["letter_ids"], rt["np"])
    return result


def infer_reasoning(model: Any, tokenizer: Any, ex: dict[str, Any], ctx: dict[str, Any],
                    args: argparse.Namespace, rt: dict[str, Any], include_hidden: bool = False) -> dict[str, Any]:
    reasoning_prompt = prompt_for(ex, reasoning=True)
    rinputs = tokenizer(reasoning_prompt, return_tensors="pt").to(model.device)
    with rt["torch"].inference_mode():
        generated = model.generate(**rinputs, max_new_tokens=args.max_reason_tokens, do_sample=False,
                                   use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id)
    continuation = generated[0, rinputs["input_ids"].shape[1]:]
    scratch = tokenizer.decode(continuation, skip_special_tokens=True)
    full = tokenizer(reasoning_prompt + scratch + "\nFinal answer:", return_tensors="pt").to(model.device)
    with rt["torch"].inference_mode():
        final = model(**full, output_hidden_states=include_hidden)
    probs = restricted_probs(final.logits[0, -1], len(ex["options"]), ctx["letter_ids"], rt["np"])
    return {"probs": probs, "generated_tokens": int(continuation.shape[0]), "scratch": scratch}


def infer_one(model: Any, tokenizer: Any, ex: dict[str, Any], ctx: dict[str, Any], args: argparse.Namespace,
              rt: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    fast = infer_fast(model, tokenizer, ex, ctx, rt)
    fast["seconds"] = time.perf_counter() - start
    # The cached reasoning path includes the initial fast pass for cascade timing.
    reasoning = infer_reasoning(model, tokenizer, ex, ctx, args, rt, include_hidden=True)
    reasoning["seconds"] = time.perf_counter() - start
    return {"fast": fast, "reasoning": reasoning}


def run_predictions(model: Any, tokenizer: Any, examples: list[dict[str, Any]], path: Path, fingerprint: str,
                    ctx: dict[str, Any], args: argparse.Namespace, rt: dict[str, Any]) -> list[dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
            if cache.get("fingerprint") == fingerprint:
                existing = {r["id"]: r for r in cache.get("rows", []) if "fast" in r and "reasoning" in r}
        except (json.JSONDecodeError, KeyError, TypeError):
            existing = {}
    rows: list[dict[str, Any]] = []
    for index, ex in enumerate(examples, 1):
        row = existing.get(ex["id"])
        if row is None:
            prediction = infer_one(model, tokenizer, ex, ctx, args, rt)
            row = {"id": ex["id"], "task": ex["task"], "label": ex["label"], "n_options": len(ex["options"]), **prediction}
        rows.append(row)
        if index % 5 == 0 or index == len(examples):
            write_json(path, {"fingerprint": fingerprint, "rows": rows})
    return rows


def nll(probs: list[list[float]], labels: list[int]) -> float:
    return float(-sum(math.log(max(float(p[y]), 1e-12)) for p, y in zip(probs, labels)) / max(1, len(labels)))


def calibration_metrics(probs: list[list[float]], labels: list[int], rt: dict[str, Any]) -> dict[str, float]:
    np = rt["np"]
    pred = np.asarray([int(np.argmax(p)) for p in probs])
    y = np.asarray(labels)
    confidence = np.asarray([max(p) for p in probs])
    correct = pred == y
    ece = 0.0
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        mask = (confidence >= low) & (confidence < high if index < 9 else confidence <= high)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    width = max(map(len, probs)) if probs else 0
    one_hot = np.zeros((len(labels), width), dtype=float)
    padded = np.zeros_like(one_hot)
    for i, (p, label) in enumerate(zip(probs, labels)):
        one_hot[i, label] = 1.0
        padded[i, :len(p)] = p
    return {"accuracy": float(correct.mean()) if len(correct) else 0.0,
            "macro_f1": float(rt["f1_score"](y, pred, average="macro", zero_division=0)) if len(y) else 0.0,
            "ece": float(ece), "nll": nll(probs, labels),
            "brier": float(np.mean(np.sum((padded - one_hot) ** 2, axis=1))) if len(labels) else 0.0}


def fit_temperature(rows: list[dict[str, Any]], mode: str, rt: dict[str, Any]) -> float:
    from scipy.optimize import minimize_scalar
    probs = [r[mode]["probs"] for r in rows]
    labels = [r["label"] for r in rows]
    if not rows:
        return 1.0
    def objective(log_temp: float) -> float:
        temp = math.exp(log_temp)
        calibrated = []
        for p in probs:
            values = rt["np"].asarray(p, dtype=float) ** (1 / temp)
            calibrated.append((values / values.sum()).tolist())
        return nll(calibrated, labels)
    return float(math.exp(minimize_scalar(objective, bounds=(-3, 3), method="bounded").x))


def apply_temperature(rows: list[dict[str, Any]], temperatures: dict[str, float], rt: dict[str, Any]) -> None:
    np = rt["np"]
    for row in rows:
        for mode in ("fast", "reasoning"):
            values = np.asarray(row[mode]["probs"], dtype=float) ** (1 / temperatures[mode])
            row[mode]["calibrated_probs"] = (values / values.sum()).tolist()


def gate_features(row: dict[str, Any], rt: dict[str, Any]) -> Any:
    np = rt["np"]
    p = np.asarray(row["fast"]["calibrated_probs"])
    sorted_p = np.sort(p)
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    hidden = np.asarray(row["fast"]["hidden"], dtype=np.float32)
    return np.concatenate(([sorted_p[-1], sorted_p[-1] - sorted_p[-2], entropy, row["n_options"]], hidden))


def fit_gate(seen_val: list[dict[str, Any]], ctx: dict[str, Any], rt: dict[str, Any]) -> tuple[Any, float, dict[str, float]]:
    np = rt["np"]
    X = np.stack([gate_features(r, rt) for r in seen_val])
    y = np.asarray([int(np.argmax(r["fast"]["calibrated_probs"]) != r["label"] and
                         np.argmax(r["reasoning"]["calibrated_probs"]) == r["label"]) for r in seen_val])
    if len(np.unique(y)) < 2:
        # If validation has no rescue/non-rescue contrast, the confidence gate remains usable.
        return None, 1.0, {"training_rows": float(len(y)), "rescue_rate": float(y.mean()) if len(y) else 0.0}
    gate = rt["make_pipeline"](rt["StandardScaler"](), rt["LogisticRegression"](
        max_iter=1000, class_weight="balanced", random_state=SEED))
    gate.fit(X, y)
    scores = gate.predict_proba(X)[:, 1]
    threshold, _ = choose_threshold(seen_val, scores, ctx, rt)
    pred = scores >= threshold
    diagnostics = {"training_rows": float(len(y)), "rescue_rate": float(y.mean()),
                   "rescue_classification_accuracy": float((pred == y).mean())}
    return gate, threshold, diagnostics


def choose_threshold(rows: list[dict[str, Any]], scores: Any, ctx: dict[str, Any], rt: dict[str, Any]) -> tuple[float, float]:
    np = rt["np"]
    scores = np.asarray(scores, dtype=float)
    labels = [r["label"] for r in rows]
    candidates = sorted(set([0.0, 1.0] + [float(x) for x in scores]))
    best_threshold, best_utility = 1.0, -float("inf")
    for threshold in candidates:
        selected = []
        for row, score in zip(rows, scores):
            mode = "reasoning" if score >= threshold else "fast"
            selected.append(row[mode]["calibrated_probs"])
        accuracy = float(np.mean([int(np.argmax(p) == y) for p, y in zip(selected, labels)])) if labels else 0.0
        rate = float(np.mean(scores >= threshold)) if len(scores) else 0.0
        utility = accuracy - GATE_ESCALATION_COST * rate
        if utility > best_utility:
            best_threshold, best_utility = threshold, utility
    return best_threshold, best_utility


def evaluate_gate(rows: list[dict[str, Any]], scores: Any, threshold: float, rt: dict[str, Any]) -> dict[str, Any]:
    np = rt["np"]
    probs: list[list[float]] = []
    labels = [r["label"] for r in rows]
    escalated: list[bool] = []
    tokens: list[int] = []
    seconds: list[float] = []
    for row, score in zip(rows, scores):
        escalate = bool(score >= threshold)
        escalated.append(escalate)
        mode = "reasoning" if escalate else "fast"
        probs.append(row[mode]["calibrated_probs"])
        tokens.append(row["reasoning"]["generated_tokens"] if escalate else 0)
        seconds.append(row[mode]["seconds"])
    result = calibration_metrics(probs, labels, rt)
    result.update({"escalation_rate": float(np.mean(escalated)) if escalated else 0.0,
                   "avg_generated_tokens": float(np.mean(tokens)) if tokens else 0.0,
                   "avg_latency_seconds": float(np.mean(seconds)) if seconds else 0.0,
                   "p50_latency_seconds": float(np.percentile(seconds, 50)) if seconds else 0.0,
                   "p95_latency_seconds": float(np.percentile(seconds, 95)) if seconds else 0.0})
    return result


def summarize_rows(rows: list[dict[str, Any]], rt: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(rows)}
    for mode in ("fast", "reasoning"):
        result[mode] = calibration_metrics([r[mode]["calibrated_probs"] for r in rows], [r["label"] for r in rows], rt)
        seconds = [r[mode]["seconds"] for r in rows]
        tokens = [r[mode]["generated_tokens"] for r in rows]
        result[mode].update({"avg_latency_seconds": float(rt["np"].mean(seconds)) if seconds else 0.0,
                             "p50_latency_seconds": float(rt["np"].percentile(seconds, 50)) if seconds else 0.0,
                             "p95_latency_seconds": float(rt["np"].percentile(seconds, 95)) if seconds else 0.0,
                             "avg_generated_tokens": float(rt["np"].mean(tokens)) if tokens else 0.0})
    return result


def make_plots(results: dict[str, Any], root: Path, rt: dict[str, Any]) -> None:
    plt = rt["plt"]
    names = list(results)
    if not names:
        return
    labels = [n.replace("_", " ") for n in names]
    metrics = [
        ("accuracy", lambda n: results[n]["test"]["overall"]["fast"]["accuracy"], "Fast accuracy"),
        ("reasoning_accuracy", lambda n: results[n]["test"]["overall"]["reasoning"]["accuracy"], "Reasoning accuracy"),
        ("trainable_parameters", lambda n: results[n]["efficiency"]["trainable_parameters"] or 0, "Trainable parameters"),
        ("checkpoint_bytes", lambda n: results[n]["efficiency"]["artifact_bytes"], "Artifact size (bytes)"),
        ("latency", lambda n: results[n]["test"]["overall"]["gate"]["avg_latency_seconds"], "Learned gate latency (s/example)"),
        ("gate_accuracy", lambda n: results[n]["test"]["overall"]["gate"]["accuracy"], "Learned gate accuracy"),
        ("confidence_gate_accuracy", lambda n: results[n]["test"]["overall"]["confidence_gate"]["accuracy"], "Confidence gate accuracy"),
    ]
    for key, getter, title in metrics:
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 4.8))
        values = [getter(name) for name in names]
        ax.bar(labels, values)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(root / "plots" / f"{key}.png", dpi=160)
        plt.close(fig)


def generate_report(results: dict[str, Any], root: Path, config: dict[str, Any]) -> None:
    lines = ["# Qwen Efficiency and Selective Reasoning Results", "",
             f"Generated by script version {SCRIPT_VERSION}. Seed: {config['seed']}. Smoke run: {config['smoke']}.", "",
             "## Overall test comparison", "",
             "| Variant | Fast acc. | Reasoning acc. | Learned gate acc. | Confidence gate acc. | Escalation | Gate tokens | Gate latency | Trainable params | Artifact MB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in results.items():
        test = item["test"]["overall"]
        eff = item["efficiency"]
        lines.append(f"| {name} | {test['fast']['accuracy']:.3f} | {test['reasoning']['accuracy']:.3f} | "
                     f"{test['gate']['accuracy']:.3f} | {test['confidence_gate']['accuracy']:.3f} | "
                     f"{test['gate']['escalation_rate']:.1%} | "
                     f"{test['gate']['avg_generated_tokens']:.1f} | {test['gate']['avg_latency_seconds']:.3f}s | "
                     f"{eff['trainable_parameters'] if eff['trainable_parameters'] is not None else 'n/a'} | "
                     f"{eff['artifact_bytes'] / (1024**2):.1f} |")
    lines += ["", "## Per-task test accuracy", "", "| Variant | Task | Fast | Reasoning | Learned gate | Confidence gate |", "|---|---|---:|---:|---:|---:|"]
    for name, item in results.items():
        for task in TASKS + ["overall"]:
            test = item["test"][task]
            lines.append(f"| {name} | {task} | {test['fast']['accuracy']:.3f} | {test['reasoning']['accuracy']:.3f} | {test['gate']['accuracy']:.3f} | {test['confidence_gate']['accuracy']:.3f} |")
    if "lora_all" in results and len(results) > 1:
        comparison_note = "Bootstrap intervals compare each variant's fast and reasoning accuracy with the `lora_all` variant."
    elif len(results) > 1:
        comparison_note = "Bootstrap intervals compare each variant with the first selected variant."
    else:
        comparison_note = "Only one variant was selected; no cross-variant bootstrap comparison is available."
    lines += ["", "## Interpretation", "",
              "Use the fast and reasoning endpoints to understand the quality/cost trade-off. The learned gate threshold was selected on validation data using utility = accuracy − 0.02 × escalation rate; test results are reported at that fixed operating point.",
              comparison_note + " A smoke run is for pipeline validation; use the full run for course conclusions.", "",
              "## Artifacts", "", "Machine-readable results are in `metrics/results.json` and `metrics/results.csv`. Comparison plots are in `plots/`.", ""]
    (root / "reports" / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_variant(name: str, spec: dict[str, Any], data: dict[str, Any], traces: list[dict[str, Any]],
                tokenizer: Any, ctx: dict[str, Any], rt: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    trace_signature = [digest(t) for t in traces] if spec["training"] != "none" else []
    fingerprint = model_fingerprint(name, spec, ctx, args, trace_signature)
    artifact_dir = ctx["root"] / "variants" / name / fingerprint
    if args.force_retrain:
        shutil.rmtree(artifact_dir / "predictions", ignore_errors=True)
    adapter_dir, training_stats = train_variant(name, spec, fingerprint, traces, data, tokenizer, ctx, rt, args)
    model = load_variant_model(name, spec, adapter_dir, ctx, rt)
    trainable = training_stats.get("trainable_parameters")
    if trainable is None:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if rt["torch"].cuda.is_available():
        rt["torch"].cuda.reset_peak_memory_stats()
    effective_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    variant_result: dict[str, Any] = {"metadata": {"variant": name, "spec": spec, "fingerprint": fingerprint},
                                      "efficiency": {**training_stats, "total_parameters": total_params,
                                                     "trainable_parameters": trainable,
                                                     "artifact_bytes": file_size(artifact_dir),
                                                     "effective_weights_bytes": effective_bytes}}
    base_examples = [r for task in TASKS for r in data[task]["val"]]
    test_examples = [r for task in TASKS for r in data[task]["test"]]
    variant_pred = ctx["root"] / "variants" / name / fingerprint / "predictions"
    val_path, test_path = variant_pred / "validation.json", variant_pred / "test.json"
    val_rows = run_predictions(model, tokenizer, base_examples, val_path, fingerprint + ":val", ctx, args, rt)
    seen_val = [r for r in val_rows if r["task"] in SEEN_TASKS]
    temperatures = {mode: fit_temperature(seen_val, mode, rt) for mode in ("fast", "reasoning")}
    apply_temperature(val_rows, temperatures, rt)
    val_dir = ctx["root"] / "variants" / name / fingerprint
    write_json(val_dir / "temperatures.json", temperatures)
    gate, learned_threshold, gate_diag = fit_gate(seen_val, ctx, rt)
    confidence_scores = [1.0 - max(r["fast"]["calibrated_probs"]) for r in seen_val]
    confidence_threshold, _ = choose_threshold(seen_val, confidence_scores, ctx, rt)
    write_json(val_dir / "gate_validation.json", {"learned_threshold": learned_threshold,
                                                   "confidence_threshold": confidence_threshold,
                                                   "diagnostics": gate_diag,
                                                   "escalation_cost": GATE_ESCALATION_COST})
    if gate is not None:
        with (val_dir / "gate.pkl").open("wb") as handle:
            pickle.dump(gate, handle)

    test_rows = run_predictions(model, tokenizer, test_examples, test_path, fingerprint + ":test", ctx, args, rt)
    apply_temperature(test_rows, temperatures, rt)
    test_groups: dict[str, list[dict[str, Any]]] = {task: [r for r in test_rows if r["task"] == task] for task in TASKS}
    test_groups["overall"] = test_rows
    val_scores = gate.predict_proba(rt["np"].stack([gate_features(r, rt) for r in seen_val]))[:, 1] if gate else confidence_scores
    if gate:
        learned_threshold, _ = choose_threshold(seen_val, val_scores, ctx, rt)
    else:
        learned_threshold = confidence_threshold
    if gate:
        write_json(val_dir / "gate_validation.json", {"learned_threshold": learned_threshold,
                                                       "confidence_threshold": confidence_threshold,
                                                       "diagnostics": gate_diag,
                                                       "escalation_cost": GATE_ESCALATION_COST})

    test_result: dict[str, Any] = {}
    for task, rows in test_groups.items():
        result = summarize_rows(rows, rt)
        learned_scores = (gate.predict_proba(rt["np"].stack([gate_features(r, rt) for r in rows]))[:, 1]
                          if gate else [1.0 - max(r["fast"]["calibrated_probs"]) for r in rows])
        confidence_scores_test = [1.0 - max(r["fast"]["calibrated_probs"]) for r in rows]
        result["gate"] = evaluate_gate(rows, learned_scores, learned_threshold, rt)
        result["confidence_gate"] = evaluate_gate(rows, confidence_scores_test, confidence_threshold, rt)
        test_result[task] = result
    variant_result["validation"] = {"temperatures": temperatures, "gate_diagnostics": gate_diag,
                                    "learned_threshold": learned_threshold, "confidence_threshold": confidence_threshold}
    variant_result["test"] = test_result
    variant_result["efficiency"]["peak_inference_gpu_bytes"] = int(rt["torch"].cuda.max_memory_allocated()) if rt["torch"].cuda.is_available() else None
    variant_result["efficiency"]["inference_examples_per_second"] = (
        len(test_rows) / sum(r["fast"]["seconds"] for r in test_rows) if test_rows else 0.0)
    baseline_rows = test_rows
    variant_result["_test_rows"] = baseline_rows
    del model
    gc.collect()
    if rt["torch"].cuda.is_available():
        rt["torch"].cuda.empty_cache()
    print(name, "test overall:", json.dumps(test_result["overall"], sort_keys=True))
    return variant_result


def main() -> None:
    args = parse_args()
    selected = [x.strip() for x in args.variants.split(",") if x.strip()]
    if not selected:
        raise SystemExit("Select at least one variant with --variants")
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Choose from: {', '.join(VARIANTS)}")
    hf_home = configure_hf_cache()
    rt = load_runtime()
    ctx = setup(args, rt)
    tokenizer = get_tokenizer(PRIMARY_MODEL, rt)
    letter_ids = get_letter_ids(tokenizer)
    ctx["letter_ids"] = letter_ids
    ctx["hf_home"] = hf_home
    write_json(ctx["root"] / "run_config.json", {**ctx["config"], "selected_variants": selected,
                                                   "letter_token_ids": letter_ids, "device": ctx["device"],
                                                   "hf_home": hf_home})
    data = load_data(ctx, rt, args)
    need_training = any(VARIANTS[name]["training"] != "none" for name in selected)
    traces = load_or_make_traces(ctx, rt, args, tokenizer, data) if need_training else []
    if not need_training:
        ctx["config"]["teacher_model_used"] = None
    write_json(ctx["root"] / "run_config.json", {**ctx["config"], "selected_variants": selected,
                                                   "letter_token_ids": letter_ids, "device": ctx["device"],
                                                   "hf_home": hf_home})
    results: dict[str, Any] = {}
    raw_test_rows: dict[str, list[dict[str, Any]]] = {}
    for name in selected:
        results[name] = run_variant(name, VARIANTS[name], data, traces, tokenizer, ctx, rt, args)
        raw_test_rows[name] = results[name].pop("_test_rows")

    reference_name = "lora_all" if "lora_all" in raw_test_rows else selected[0]
    for name, rows in raw_test_rows.items():
        results[name]["comparisons_vs_reference"] = {
            "reference_variant": reference_name,
            "fast_accuracy_difference": bootstrap_accuracy_difference_pair(rows, raw_test_rows[reference_name], "fast", rt),
            "reasoning_accuracy_difference": bootstrap_accuracy_difference_pair(rows, raw_test_rows[reference_name], "reasoning", rt),
        }

    # Build compact machine-readable summaries; retain full sample predictions separately per variant.
    public_results = {name: {k: v for k, v in result.items()} for name, result in results.items()}
    write_json(ctx["root"] / "metrics" / "results.json", {"config": ctx["config"], "results": public_results})
    rows_for_csv = []
    for name, item in results.items():
        for task, values in item["test"].items():
            row = {"variant": name, "task": task, "n": values["n"]}
            for mode in ("fast", "reasoning", "gate", "confidence_gate"):
                for metric, value in values[mode].items():
                    row[f"{mode}_{metric}"] = value
            row.update({f"efficiency_{k}": v for k, v in item["efficiency"].items()})
            rows_for_csv.append(row)
    rt["pd"].DataFrame(rows_for_csv).to_csv(ctx["root"] / "metrics" / "results.csv", index=False)
    make_plots(results, ctx["root"], rt)
    generate_report(results, ctx["root"], ctx["config"])
    print(f"Saved results and report under {ctx['root']}")


def bootstrap_accuracy_difference_pair(rows: list[dict[str, Any]], reference: list[dict[str, Any]], mode: str,
                                       rt: dict[str, Any], repetitions: int = 2000) -> dict[str, float]:
    np = rt["np"]
    ref_by_id = {r["id"]: r for r in reference}
    paired = [(r, ref_by_id[r["id"]]) for r in rows if r["id"] in ref_by_id]
    if not paired:
        return {"mean_difference": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    differences = []
    for row, ref in paired:
        label = row["label"]
        target = int(np.argmax(row[mode]["calibrated_probs"]) == label)
        baseline = int(np.argmax(ref[mode]["calibrated_probs"]) == ref["label"])
        differences.append(target - baseline)
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = [float(np.mean(differences[rng.integers(0, len(differences), len(differences))])) for _ in range(repetitions)]
    return {"mean_difference": float(differences.mean()), "ci95_low": float(np.percentile(samples, 2.5)),
            "ci95_high": float(np.percentile(samples, 97.5))}


if __name__ == "__main__":
    main()
