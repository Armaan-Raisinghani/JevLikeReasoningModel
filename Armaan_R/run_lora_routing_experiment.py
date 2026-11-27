#!/usr/bin/env python3
"""Train a larger all-projection LoRA adapter and compare routing against job 17248."""

import argparse
import json
import random
from pathlib import Path

import evaluate_frozen_routing as routing
import train_jev_efficiency_experiments as jev


VARIANT = "lora_all"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("traces", "train", "evaluate"))
    parser.add_argument("--reference-dir", type=Path, required=True, help="Output directory of frozen job 17248.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Exercise the stages on small subsets.")
    args = parser.parse_args()
    args.seed = jev.SEED
    args.learning_rate = 2e-4
    args.force_retrain = False
    return args


def reference_data(root):
    reference = json.loads((root / "run_config.json").read_text())
    if reference["model"] != jev.PRIMARY_MODEL or reference["seed"] != jev.SEED:
        raise ValueError("Expected a frozen Qwen3-1.7B reference using the experiment seed.")
    split_paths = sorted((root / "data").glob("splits_*.json"))
    if len(split_paths) != 1:
        raise ValueError("Expected exactly one reference split file.")
    data = json.loads(split_paths[0].read_text())
    predictions = json.loads((root / "metrics" / "fast_only_predictions.json").read_text())
    original_test = [r for task in jev.TASKS for r in data[task]["test"]]
    if [r["id"] for r in original_test] != [r["id"] for r in predictions]:
        raise ValueError("Reference splits do not match the frozen job's scored examples.")
    validation = [r for task in jev.SEEN_TASKS for r in data[task]["val"]]
    scored_validation = json.loads((root / "validation_predictions.json").read_text())["rows"]
    if [r["id"] for r in validation] != [r["id"] for r in scored_validation]:
        raise ValueError("Reference splits do not match the frozen job's validation examples.")
    return reference, data


def question_key(row):
    return " ".join(row["question"].split()).casefold()


def prepare_training(data, ctx, args, rt):
    held_out = {question_key(row) for task in jev.TASKS
                for split in ("val", "test") for row in data[task][split]}
    # Word math stays held out, including when the smoke run subsets its evaluation.
    source_data = jev.load_data({**ctx, "root": ctx["root"] / "training_pool"}, rt, args)
    direct, eligible = [], []
    for task in jev.SEEN_TASKS:
        unique = {}
        for row in source_data[task]["train"]:
            key = question_key(row)
            if key not in held_out:
                unique.setdefault(key, row)
        pool = list(unique.values())
        data[task]["train"] = pool
        direct.extend(pool[:ctx["config"]["direct_per_task"]])
        eligible.extend(pool)
    random.Random(args.seed).shuffle(eligible)
    return direct, eligible[:ctx["config"]["max_trace_attempts"]]


def generate_traces(path, key, eligible, tokenizer, ctx, rt):
    state = json.loads(path.read_text()) if path.exists() else {"key": key, "attempted": 0, "traces": []}
    if state["key"] != key:
        raise ValueError("Trace cache belongs to a different configuration. Use a fresh output directory.")
    target = ctx["config"]["target_accepted_traces"]
    if len(state["traces"]) >= target or state["attempted"] == len(eligible):
        return
    teacher = jev.model_load(jev.PRIMARY_MODEL, ctx, rt).eval()
    for index, example in enumerate(eligible[state["attempted"]:], state["attempted"] + 1):
        trace = jev.generate_teacher_trace(teacher, tokenizer, example, rt)
        if trace is not None:
            state["traces"].append(trace)
        state["attempted"] = index
        done = index == len(eligible) or len(state["traces"]) >= target
        if done or index % 10 == 0:
            jev.write_json(path, state)
            print(f"Trace attempts {index}/{len(eligible)}, accepted {len(state['traces'])}", flush=True)
        if done:
            break


def main():
    args = parse_args()
    hf_home = jev.configure_hf_cache()
    rt = jev.load_runtime()
    torch = rt["torch"]
    if not torch.cuda.is_available():
        raise SystemExit("This experiment requires a CUDA GPU.")
    reference, data = reference_data(args.reference_dir)
    args.max_reason_tokens = reference["max_reason_tokens"]
    args.target_escalation = reference["target_escalation"]
    args.accuracy_gap = reference["accuracy_gap"]
    ctx = jev.setup(args, rt)
    ctx["config"].update(
        epochs=1 if args.smoke else 2, train_per_task=40 if args.smoke else 2000,
        val_per_task=100, test_per_task=100,
        direct_per_task=20 if args.smoke else 270,
        target_accepted_traces=20 if args.smoke else 1500,
        max_trace_attempts=40 if args.smoke else 3000,
    )
    baseline_results = json.loads((args.reference_dir / "metrics" / "routing_results.json").read_text())
    target_seconds = baseline_results["results"]["raw_confidence_escalation_budget"]["overall"]["mean_seconds"]

    from transformers import AutoConfig
    model_revision = AutoConfig.from_pretrained(jev.PRIMARY_MODEL)._commit_hash
    if not model_revision or model_revision != reference["model_revision"]:
        raise RuntimeError("Cached base model revision differs from the frozen reference.")
    ctx["config"].update(model_revision=model_revision, teacher_model_used=jev.PRIMARY_MODEL,
                         reference_split_digest=jev.digest(data))
    tokenizer = jev.get_tokenizer(jev.PRIMARY_MODEL, rt)
    ctx["letter_ids"] = jev.get_letter_ids(tokenizer)
    root = ctx["root"]
    trace_path = root / "traces" / "progress.json"
    adapter_record = root / "trained_adapter.json"

    if args.stage != "evaluate":
        direct, eligible = prepare_training(data, ctx, args, rt)
        key = jev.digest([ctx["config"], eligible, ctx["letter_ids"]])
        direct_ids = {r["id"] for r in direct}
        if not direct_ids or not eligible:
            raise RuntimeError("No eligible training data.")
        if args.stage == "traces":
            generate_traces(trace_path, key, eligible, tokenizer, ctx, rt)
            return
        state = json.loads(trace_path.read_text())
        traces = state["traces"]
        if state["key"] != key or (
            state["attempted"] != len(eligible) and len(traces) < ctx["config"]["target_accepted_traces"]
        ):
            raise RuntimeError("Finish the matching traces stage before training.")
        if not traces:
            raise RuntimeError("No teacher traces passed the answer filter.")
        fingerprint = jev.model_fingerprint(
            VARIANT, jev.VARIANTS[VARIANT], ctx, args, [jev.digest([sorted(direct_ids), traces])]
        )
        artifact = root / "variants" / VARIANT / fingerprint
        args.force_retrain = not (
            (artifact / "manifest.json").is_file() and
            (artifact / "adapter" / "adapter_model.safetensors").is_file()
        )
        adapter, stats = jev.train_variant(
            VARIANT, jev.VARIANTS[VARIANT], fingerprint, traces, data, tokenizer, ctx, rt, args,
            direct_ids=direct_ids,
        )
        jev.write_json(adapter_record, {"path": str(adapter), "fingerprint": fingerprint,
                                       "training_config": ctx["config"], "training_stats": stats,
                                       "direct_examples": len(direct), "trace_attempts": state["attempted"],
                                       "accepted_traces": len(traces)})
        return

    if args.smoke:
        for task in jev.TASKS:
            data[task]["val"] = data[task]["val"][:20]
            data[task]["test"] = data[task]["test"][:20]
    validation = [r for task in jev.SEEN_TASKS for r in data[task]["val"]]
    test = [r for task in jev.TASKS for r in data[task]["test"]]
    adapter = json.loads(adapter_record.read_text())
    if adapter["training_config"] != ctx["config"]:
        raise ValueError("Training configuration does not match. Use the same options in each stage.")
    if ctx["precision"] != reference["precision"]:
        raise RuntimeError("Adapter evaluation precision differs from the frozen reference.")
    if torch.cuda.get_device_name() != reference["gpu"]:
        print("GPU differs from the frozen reference. Compare the latency results with that difference in mind.",
              flush=True)
    model = jev.load_variant_model(VARIANT, jev.VARIANTS[VARIANT], Path(adapter["path"]), ctx, rt)
    config = {**ctx["config"], "model": jev.PRIMARY_MODEL, "hf_home": hf_home,
              "gpu": torch.cuda.get_device_name(), "reference_dir": str(args.reference_dir.resolve()),
              "target_mean_seconds": target_seconds, "validation_examples": len(validation),
              "test_examples": len(test), "adapter": adapter}
    jev.write_json(root / "run_config.json", config)
    routing.evaluate(model, tokenizer, validation, test, ctx, args, rt, config,
                     checkpoint_id=adapter["fingerprint"], gates=("raw_confidence", "learned_rescue"),
                     operating_points=("escalation_budget", "latency_target"), target_seconds=target_seconds)


if __name__ == "__main__":
    main()
