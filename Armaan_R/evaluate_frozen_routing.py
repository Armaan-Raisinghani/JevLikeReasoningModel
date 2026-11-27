#!/usr/bin/env python3
"""Compare validation-fixed routing policies on frozen Qwen3-1.7B."""

import argparse
import time

import train_jev_efficiency_experiments as jev


GATES = ("raw_confidence", "calibrated_confidence", "learned_rescue")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true", help="Use 20 validation and test questions per task.")
    parser.add_argument("--val-per-task", type=int, default=100, help="Validation count per seen task.")
    parser.add_argument("--test-per-task", type=int, default=100, help="Test count per task.")
    parser.add_argument("--seed", type=int, default=jev.SEED)
    parser.add_argument("--max-reason-tokens", type=int, default=256)
    parser.add_argument("--target-escalation", type=float, default=0.24)
    parser.add_argument("--accuracy-gap", type=float, default=0.03)
    args = parser.parse_args()
    if not 0 <= args.target_escalation <= 1 or not 0 <= args.accuracy_gap <= 1:
        parser.error("Escalation and accuracy gap must be fractions between 0 and 1.")
    if min(args.val_per_task, args.test_per_task) < 1:
        parser.error("Validation and test counts must be positive.")
    if args.max_reason_tokens < 1:
        parser.error("--max-reason-tokens must be positive.")
    return args


def fit_rescue(rows, rt, seed):
    np = rt["np"]
    labels = [int(np.argmax(row["fast"]["probs"]) != row["label"]
                  and np.argmax(row["reasoning"]["probs"]) == row["label"]) for row in rows]
    if len(set(labels)) < 2:
        return None
    gate = rt["make_pipeline"](
        rt["StandardScaler"](),
        rt["LogisticRegression"](max_iter=1000, class_weight="balanced", random_state=seed),
    )
    return gate.fit(rt["np"].stack([jev.gate_features(row, rt) for row in rows]), labels)


def gate_scores(name, rows, gate, rt):
    if name == "learned_rescue":
        return gate.predict_proba(rt["np"].stack([jev.gate_features(row, rt) for row in rows]))[:, 1]
    key = "probs" if name == "raw_confidence" else "calibrated_probs"
    return [1.0 - max(row["fast"][key]) for row in rows]


def choose_operating_points(rows, scores, budget, gap, np):
    scores = np.asarray(scores, dtype=float)
    fast = np.asarray([np.argmax(r["fast"]["probs"]) == r["label"] for r in rows], dtype=int)
    reason = np.asarray([np.argmax(r["reasoning"]["probs"]) == r["label"] for r in rows], dtype=int)
    candidates = []
    # A finite threshold above every probability allows the no-escalation policy.
    for threshold in sorted(set(scores.tolist() + [2.0])):
        route = scores >= threshold
        candidates.append({"threshold": threshold, "validation_escalation": float(route.mean()),
                           "validation_accuracy": float(np.where(route, reason, fast).mean())})
    at_budget = [p for p in candidates if p["validation_escalation"] <= budget]
    near_reason = [p for p in candidates if p["validation_accuracy"] >= reason.mean() - gap - 1e-12]
    return {
        "escalation_budget": max(at_budget, key=lambda p: (p["validation_accuracy"], -p["validation_escalation"])),
        "accuracy_gap": min(near_reason, key=lambda p: (p["validation_escalation"], -p["validation_accuracy"])),
    }



def time_policy(name, threshold, examples, model, tokenizer, ctx, args, rt, gate, temperature):
    results = []
    for i, ex in enumerate(examples, 1):
        torch, np = rt["torch"], rt["np"]
        torch.cuda.synchronize()
        start = time.perf_counter()
        escalate = name == "always_reason"
        if not escalate:
            fast = jev.infer_fast(model, tokenizer, ex, ctx, rt, include_hidden=name == "learned_rescue")
            result = fast
            if name in GATES:
                if name != "raw_confidence":
                    p = np.asarray(fast["probs"], dtype=float) ** (1 / temperature)
                    fast["calibrated_probs"] = (p / p.sum()).tolist()
                row = {"fast": fast, "n_options": len(ex["options"])}
                escalate = bool(gate_scores(name, [row], gate, rt)[0] >= threshold)
        if escalate:
            result = jev.infer_reasoning(model, tokenizer, ex, ctx, args, rt)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        results.append({"id": ex["id"], "task": ex["task"], "label": ex["label"],
                "prediction": int(np.argmax(result["probs"])), "escalated": escalate,
                "generated_tokens": result["generated_tokens"], "seconds": seconds})
        if i % 100 == 0 or i == len(examples):
            print(f"  {i}/{len(examples)} questions timed", flush=True)
    return results


def summarize(rows, np):
    seconds = [r["seconds"] for r in rows]
    return {"n": len(rows), "accuracy": float(np.mean([r["prediction"] == r["label"] for r in rows])),
            "escalation_rate": float(np.mean([r["escalated"] for r in rows])),
            "mean_generated_tokens": float(np.mean([r["generated_tokens"] for r in rows])),
            "mean_seconds": float(np.mean(seconds)), "p95_seconds": float(np.percentile(seconds, 95))}


def main():
    args = parse_args()
    hf_home = jev.configure_hf_cache()
    rt = jev.load_runtime()
    torch = rt["torch"]
    if not torch.cuda.is_available():
        raise SystemExit("Run this evaluation on a CUDA GPU for comparable latency measurements.")

    # The shared setup and data loader also serve the training pipeline.
    args.learning_rate = None
    ctx = jev.setup(args, rt)
    ctx["config"].update(train_per_task=0, val_per_task=20 if args.smoke else args.val_per_task,
                         test_per_task=20 if args.smoke else args.test_per_task)
    tokenizer = jev.get_tokenizer(jev.PRIMARY_MODEL, rt)
    model = jev.model_load(jev.PRIMARY_MODEL, ctx, rt).eval()
    ctx["letter_ids"] = jev.get_letter_ids(tokenizer)

    data = jev.load_data(ctx, rt, args)
    validation = [r for task in jev.SEEN_TASKS for r in data[task]["val"]]
    test = [r for task in jev.TASKS for r in data[task]["test"]]
    if not validation or not test:
        raise RuntimeError("Validation and test must each contain usable examples.")
    if {r["id"] for r in validation} & {r["id"] for r in test}:
        raise RuntimeError("Validation and test examples overlap.")
    config = {"model": jev.PRIMARY_MODEL, "model_revision": getattr(model.config, "_commit_hash", None),
              "seed": args.seed, "smoke": args.smoke, "max_reason_tokens": args.max_reason_tokens,
              "target_escalation": args.target_escalation, "accuracy_gap": args.accuracy_gap,
              "validation_tasks": jev.SEEN_TASKS, "validation_examples": len(validation),
              "test_examples": len(test), "precision": ctx["precision"],
              "gpu": torch.cuda.get_device_name(), "hf_home": hf_home}
    root = ctx["root"]
    jev.write_json(root / "run_config.json", config)
    print(f"Scoring {len(validation)} seen-task validation questions", flush=True)
    fingerprint = jev.digest([jev.SCRIPT_VERSION, config["model"], config["model_revision"], ctx["precision"],
                              args.max_reason_tokens, ctx["letter_ids"], validation])
    rows = jev.run_predictions(model, tokenizer, validation, root / "validation_predictions.json",
                               fingerprint, ctx, args, rt)
    temperatures = {mode: jev.fit_temperature(rows, mode, rt) for mode in ("fast", "reasoning")}
    jev.apply_temperature(rows, temperatures, rt)
    gate = fit_rescue(rows, rt, args.seed)
    if gate is None:
        print("Learned rescue gate unavailable: validation needs rescue and non-rescue examples.", flush=True)
    policies = {"fast_only": ("fast_only", None), "always_reason": ("always_reason", None)}
    for name in GATES:
        if name == "learned_rescue" and gate is None:
            continue
        scores = gate_scores(name, rows, gate, rt)
        points = choose_operating_points(rows, scores, args.target_escalation, args.accuracy_gap, rt["np"])
        for target, point in points.items():
            policies[f"{name}_{target}"] = (name, point)

    # Warm both paths on validation examples so every test question remains in the metrics.
    for ex in validation[:3]:
        jev.infer_fast(model, tokenizer, ex, ctx, rt)
        jev.infer_reasoning(model, tokenizer, ex, ctx, args, rt)
    results = {}
    timed = {}
    for name, (method, point) in policies.items():
        print(f"Timing {name}", flush=True)
        key = (method, point["threshold"] if point else None)
        if key not in timed:
            timed[key] = time_policy(*key, test, model, tokenizer, ctx, args, rt, gate, temperatures["fast"])
        predictions = timed[key]
        jev.write_json(root / "metrics" / f"{name}_predictions.json", predictions)
        overall = summarize(predictions, rt["np"])
        results[name] = {"selection": point, "overall": overall,
                         "by_task": {task: summarize(group, rt["np"]) for task in jev.TASKS
                                     if (group := [r for r in predictions if r["task"] == task])}}
        jev.write_json(root / "metrics" / "routing_results.json",
                       {"config": config, "temperatures": temperatures, "results": results})
        print(f"{name}: {overall['accuracy']:.2%} accuracy, {overall['escalation_rate']:.2%} escalated, "
              f"{overall['mean_seconds']:.3f} s mean, {overall['p95_seconds']:.3f} s p95", flush=True)


if __name__ == "__main__":
    main()
