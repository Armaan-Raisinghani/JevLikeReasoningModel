#!/usr/bin/env python3
"""Measure frozen, unmerged, and merged LoRA on the same GPU and questions."""

import argparse
import json
import os
from pathlib import Path

import evaluate_frozen_routing as routing
import run_lora_routing_experiment as lora
import train_jev_efficiency_experiments as jev


MODES = ("fast_only", "always_reason", "raw_confidence")
VARIANTS = ("frozen", "unmerged_lora", "merged_lora")
RAW_BUDGET = "raw_confidence_escalation_budget"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Time ten test questions per task.")
    args = parser.parse_args()
    args.learning_rate = None
    return args


def read_inputs(reference_dir, lora_dir):
    reference, data = lora.reference_data(reference_dir)
    adapter = json.loads((lora_dir / "trained_adapter.json").read_text())
    lora_results = json.loads((lora_dir / "metrics" / "routing_results.json").read_text())
    if adapter["fingerprint"] != lora_results["config"]["adapter"]["fingerprint"]:
        raise ValueError("LoRA results and adapter record refer to different checkpoints.")
    if adapter["training_config"]["reference_split_digest"] != jev.digest(data):
        raise ValueError("LoRA adapter and frozen run do not use the same reference splits.")
    for key in ("model_revision", "max_reason_tokens"):
        if adapter["training_config"][key] != reference[key]:
            raise ValueError(f"LoRA adapter and frozen run disagree on {key}.")
    test = [r for task in jev.TASKS for r in data[task]["test"]]
    lora_test = json.loads((lora_dir / "metrics" / "fast_only_predictions.json").read_text())
    if [(r["id"], r["label"]) for r in test] != [(r["id"], r["label"]) for r in lora_test]:
        raise ValueError("The previous LoRA run scored different test questions.")
    frozen_results = json.loads((reference_dir / "metrics" / "routing_results.json").read_text())
    frozen_threshold = frozen_results["results"][RAW_BUDGET]["selection"]["threshold"]
    lora_threshold = lora_results["results"][RAW_BUDGET]["selection"]["threshold"]
    thresholds = {"frozen": frozen_threshold, "unmerged_lora": lora_threshold, "merged_lora": lora_threshold}
    warmup = [r for task in jev.SEEN_TASKS for r in data[task]["val"]][:3]
    return reference, adapter, data, warmup, thresholds


def benchmark_mode(mode, test, models, tokenizer, ctx, args, rt, thresholds, fingerprint):
    path = ctx["root"] / "predictions" / f"{mode}.json"
    state = json.loads(path.read_text()) if path.exists() else {"fingerprint": fingerprint, "rows": []}
    if state["fingerprint"] != fingerprint:
        raise ValueError(f"Existing timings do not match this benchmark: {path}")
    rows = state["rows"]
    for index in range(len(rows), len(test)):
        ex = test[index]
        # Rotate the execution order to distribute any first/last timing effect.
        offset = index % len(VARIANTS)
        order = VARIANTS[offset:] + VARIANTS[:offset]
        measured = {}
        for variant in order:
            threshold = thresholds[variant] if mode == "raw_confidence" else None
            measured[variant] = routing.time_request(
                mode, threshold, ex, models[variant], tokenizer, ctx, args, rt, None, 1.0
            )
        rows.append({"job_id": os.environ.get("SLURM_JOB_ID"), "models": measured})
        if (index + 1) % 10 == 0 or index + 1 == len(test):
            jev.write_json(path, state)
        if (index + 1) % 25 == 0 or index + 1 == len(test):
            print(f"{mode}: {index + 1}/{len(test)} questions timed for all three models", flush=True)
    return rows


def summarize_mode(rows, np):
    by_model = {v: [row["models"][v] for row in rows] for v in VARIANTS}
    unmerged, merged = by_model["unmerged_lora"], by_model["merged_lora"]
    return {
        "models": {v: {"overall": routing.summarize(items, np),
                       "by_task": {task: routing.summarize(group, np) for task in jev.TASKS
                                   if (group := [row for row in items if row["task"] == task])}}
                   for v, items in by_model.items()},
        "merge_agreement": {
            "same_answer_fraction": float(np.mean([a["prediction"] == b["prediction"] for a, b in zip(unmerged, merged)])),
            "same_route_fraction": float(np.mean([a["escalated"] == b["escalated"] for a, b in zip(unmerged, merged)])),
        },
        "slurm_job_ids": sorted({row["job_id"] for row in rows if row["job_id"] is not None}),
    }


def main():
    args = parse_args()
    hf_home = jev.configure_hf_cache()
    rt = jev.load_runtime()
    torch = rt["torch"]
    if not torch.cuda.is_available():
        raise SystemExit("Run the benchmark on a CUDA GPU.")

    reference, adapter, data, warmup, thresholds = read_inputs(args.reference_dir, args.lora_dir)
    args.seed = reference["seed"]
    args.max_reason_tokens = reference["max_reason_tokens"]
    test = [r for task in jev.TASKS for r in (data[task]["test"][:10] if args.smoke else data[task]["test"])]
    ctx = jev.setup(args, rt)
    if ctx["precision"] != reference["precision"]:
        raise ValueError("GPU precision differs from the original experiment.")
    tokenizer = jev.get_tokenizer(jev.PRIMARY_MODEL, rt)
    ctx["letter_ids"] = jev.get_letter_ids(tokenizer)
    adapter_path = Path(adapter["path"])
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Adapter weights missing from {adapter_path}")

    frozen = jev.model_load(jev.PRIMARY_MODEL, ctx, rt).eval()
    if frozen.config._commit_hash != reference["model_revision"]:
        raise ValueError("Cached base model revision differs from the reference.")
    unmerged = jev.load_variant_model(lora.VARIANT, jev.VARIANTS[lora.VARIANT], adapter_path, ctx, rt)
    merge_source = jev.load_variant_model(lora.VARIANT, jev.VARIANTS[lora.VARIANT], adapter_path, ctx, rt)
    with torch.no_grad():
        merged = merge_source.merge_and_unload(safe_merge=True).eval()
    del merge_source
    models = {"frozen": frozen, "unmerged_lora": unmerged, "merged_lora": merged}
    for variant, model in models.items():
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise RuntimeError(f"{variant} is partially offloaded. The timing comparison requires all weights on GPU.")
        for ex in warmup:
            jev.infer_fast(model, tokenizer, ex, ctx, rt, include_hidden=False)
            jev.infer_reasoning(model, tokenizer, ex, ctx, args, rt)

    config = {"model": jev.PRIMARY_MODEL, "model_revision": reference["model_revision"],
              "adapter": adapter["fingerprint"], "precision": ctx["precision"],
              "gpu": torch.cuda.get_device_name(), "torch_version": str(torch.__version__),
              "max_reason_tokens": args.max_reason_tokens, "hf_home": hf_home,
              "test_examples": len(test), "test_digest": jev.digest(test), "raw_confidence_thresholds": thresholds}
    root = ctx["root"]
    jev.write_json(root / "run_config.json", config)
    results = {}
    for mode in MODES:
        print(f"Timing {mode}", flush=True)
        rows = benchmark_mode(mode, test, models, tokenizer, ctx, args, rt, thresholds,
                              jev.digest([config, mode, ctx["letter_ids"]]))
        results[mode] = summarize_mode(rows, rt["np"])
        jev.write_json(root / "metrics" / "merged_lora_results.json", {"config": config, "results": results})
        for variant, result in results[mode]["models"].items():
            m = result["overall"]
            print(f"{variant}: {m['accuracy']:.2%} accuracy, {m['escalation_rate']:.2%} escalated, "
                  f"{m['mean_seconds']:.3f} s mean, {m['p95_seconds']:.3f} s p95", flush=True)


if __name__ == "__main__":
    main()
