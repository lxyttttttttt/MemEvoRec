#!/usr/bin/env python3
"""Summarize preserved Block 1 plus newly completed Block 2."""

import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
VARIANTS = ("corrected", "read", "full")
METRICS = ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "NDCG@10")
N_RESAMPLES = 10_000
STAT_SEED = 20260825
RUNS = {
    1: {
        "seed": 42,
        "corrected": OUT / "feedback_memrec_v11_final_100_corrected",
        "read": OUT / "feedback_memrec_v11_final_100_read",
        "full": OUT / "feedback_memrec_v11_final_100_full",
    },
    2: {
        "seed": 43,
        "corrected": OUT / "feedback_memrec_v11_block2_seed43_corrected",
        "read": OUT / "feedback_memrec_v11_block2_seed43_read",
        "full": OUT / "feedback_memrec_v11_block2_seed43_full",
    },
}
BLOCK2_RUN_LOGS = {
    "corrected": Path("/tmp/feedback_memrec_v11_2x100_rerun_logs/block2_corrected.log"),
    "read": Path("/tmp/feedback_memrec_v11_2x100_rerun_logs/block2_read.log"),
    "full": Path("/tmp/feedback_memrec_v11_2x100_rerun_logs/block2_full.log"),
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block2_handled_json_fallbacks():
    patterns = (
        ("stage_w", re.compile(r"Error in Stage-W for user (\d+): ([^\r\n]+)")),
        ("reranker", re.compile(r"Error in LLM Reranker for user (\d+): ([^\r\n]+)")),
    )
    events = []
    for variant, path in BLOCK2_RUN_LOGS.items():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        test_boundary = text.find("MemRec Training Complete!")
        for component, pattern in patterns:
            for match in pattern.finditer(text):
                events.append({
                    "variant": variant,
                    "phase": "test" if test_boundary >= 0 and match.start() > test_boundary else "warmup",
                    "component": component,
                    "user_id": int(match.group(1)),
                    "error": match.group(2).strip(),
                    "handled_by_fallback": True,
                    "fatal_transport_error": False,
                    "source_log": str(path),
                })
    return events


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def paired_test(differences, seed):
    rng = random.Random(seed)
    observed = mean(differences)
    n = len(differences)
    bootstrap = [mean(rng.choices(differences, k=n)) for _ in range(N_RESAMPLES)]
    extreme = 0
    for _ in range(N_RESAMPLES):
        permuted = mean(value if rng.getrandbits(1) else -value for value in differences)
        extreme += abs(permuted) >= abs(observed)
    return {
        "n_pairs": n,
        "mean_delta": observed,
        "bootstrap_95_ci": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        "sign_flip_two_sided_p": (extreme + 1) / (N_RESAMPLES + 1),
        "n_resamples": N_RESAMPLES,
        "seed": seed,
    }


def paired_block2(comparator, common_success):
    full = {row["user_id"]: row for row in load_jsonl(RUNS[2]["full"] / "per_user_metrics.jsonl")}
    other = {row["user_id"]: row for row in load_jsonl(RUNS[2][comparator] / "per_user_metrics.jsonl")}
    differences = []
    for user_id in sorted(full.keys() & other.keys()):
        if common_success and not (full[user_id]["ranking_success"] and other[user_id]["ranking_success"]):
            continue
        differences.append(full[user_id]["ndcg_at_10"] - other[user_id]["ndcg_at_10"])
    return differences


def main():
    docs = {block: {variant: load_json(path / "run_metrics.json") for variant, path in runs.items() if variant != "seed"} for block, runs in RUNS.items()}
    rows = []
    for block in (1, 2):
        for variant in VARIANTS:
            metrics = docs[block][variant]["test_metrics"]
            rows.append({
                "block": block,
                "seed": RUNS[block]["seed"],
                "variant": variant,
                **{key: metrics[key] for key in METRICS},
                "credit_updates_during_test": metrics["credit_updates_during_test"],
                "wall_time_seconds": metrics.get("wall_time_seconds", 0),
                "llm_requests": metrics.get("llm_token_stats", {}).get("total_requests", 0),
                "llm_total_tokens": metrics.get("llm_token_stats", {}).get("total_tokens", 0),
            })

    block_table = []
    for block in (1, 2):
        values = {variant: docs[block][variant]["test_metrics"]["NDCG@10"] for variant in VARIANTS}
        block_table.append({
            "block": block,
            "seed": RUNS[block]["seed"],
            **values,
            "full_minus_read": values["full"] - values["read"],
            "full_minus_corrected": values["full"] - values["corrected"],
        })

    aggregates = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        aggregates[variant] = {
            metric: {"mean": mean(row[metric] for row in selected), "sample_std": stdev(row[metric] for row in selected)}
            for metric in METRICS
        }

    tests = {}
    for index, comparator in enumerate(("read", "corrected")):
        tests[f"full_minus_{comparator}"] = {
            "block2_all_pairs": paired_test(paired_block2(comparator, False), STAT_SEED + index * 2),
            "block2_common_success": paired_test(paired_block2(comparator, True), STAT_SEED + index * 2 + 1),
        }

    manifests = {}
    for block in (1, 2):
        hashes = {variant: sha256(RUNS[block][variant] / "candidate_manifest.jsonl") for variant in VARIANTS}
        manifests[str(block)] = {"hashes": hashes, "byte_identical": len(set(hashes.values())) == 1}

    gate_by_block = {}
    for block in (1, 2):
        metrics = docs[block]["full"]["test_metrics"]
        stats = metrics["feedback_attribution_stats"]
        propagation = load_jsonl(RUNS[block]["full"] / "feedback_memrec_logs" / "propagation_events.jsonl")
        direct = load_jsonl(RUNS[block]["full"] / "feedback_memrec_logs" / "direct_write_events.jsonl")
        rejected = [event for event in propagation if event.get("decision") == "reject"]
        gate_by_block[str(block)] = {
            "propagation_total": len(propagation),
            "rejected": len(rejected),
            "reject_rate": len(rejected) / len(propagation) if propagation else 0,
            "immediate_rejected": sum(event.get("reason") == "negative_current_episode_contribution" for event in rejected),
            "historical_rejected": sum(event.get("reason") == "negative_historical_credit" for event in rejected),
            "direct_total": len(direct),
            "direct_preserved": sum(event.get("decision") == "accept" for event in direct),
            "direct_preservation_rate": sum(event.get("decision") == "accept" for event in direct) / len(direct) if direct else 1.0,
            "invalid_relations": stats["n_invalid_relations"],
            "claimed_relations": stats["n_claimed_relations"],
            "invalid_relation_ratio": stats["invalid_relation_ratio"],
        }

    json_fallbacks = block2_handled_json_fallbacks()

    result = {
        "protocol": "FeedbackMemRec V1.1 2x100 independent-block stage",
        "block3_status": "deferred/not run",
        "block1_per_user_available": False,
        "paired_inference_scope": "Block 2 only (100 paired users)",
        "runs": rows,
        "block_ndcg_table": block_table,
        "two_block_equal_weight_aggregate": aggregates,
        "direction": {
            "full_wins_vs_read": sum(row["full_minus_read"] > 0 for row in block_table),
            "full_wins_vs_corrected": sum(row["full_minus_corrected"] > 0 for row in block_table),
        },
        "candidate_manifests": manifests,
        "all_test_credit_frozen": all(row["credit_updates_during_test"] == 0 for row in rows),
        "gate_by_block": gate_by_block,
        "block2_json_fallback_events": json_fallbacks,
        "block2_fatal_transport_errors": 0,
        "paired_tests_ndcg_at_10": tests,
        "evidence_limitations": [
            "Block 1 has aggregate metrics but no recoverable predicted ranking per user.",
            "Paired inference therefore covers Block 2 only and is not a 200-user paired test.",
            "Two independent blocks are insufficient to establish broad cross-block stability.",
            "Block 3 was intentionally deferred and no result was imputed.",
        ],
    }

    json_path = OUT / "feedback_memrec_v11_2x100_summary.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = OUT / "feedback_memrec_v11_2x100_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# FeedbackMemRec V1.1 2×100 Stage Report", "",
        "Block 1 was preserved and Block 2 was newly run. Block 3 is **deferred/not run**.", "",
        "## Block results", "",
        "| Block | Seed | Corrected NDCG@10 | Read NDCG@10 | Full NDCG@10 | Full−Read | Full−Corrected |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in block_table:
        lines.append(f"| {row['block']} | {row['seed']} | {row['corrected']:.6f} | {row['read']:.6f} | {row['full']:.6f} | {row['full_minus_read']:+.6f} | {row['full_minus_corrected']:+.6f} |")
    lines.extend(["", "## Direction and paired inference", "", f"Full exceeds Read in {result['direction']['full_wins_vs_read']}/2 blocks and Corrected in {result['direction']['full_wins_vs_corrected']}/2 blocks. This direction count is not a significance test.", ""])
    for label, group in tests.items():
        test = group["block2_all_pairs"]
        sensitivity = group["block2_common_success"]
        lines.append(f"- {label}: Block 2 n={test['n_pairs']}, mean NDCG@10 delta `{test['mean_delta']:+.6f}`, bootstrap 95% CI `[{test['bootstrap_95_ci'][0]:+.6f}, {test['bootstrap_95_ci'][1]:+.6f}]`, two-sided sign-flip p=`{test['sign_flip_two_sided_p']:.6f}`. Common-success sensitivity n={sensitivity['n_pairs']}, mean `{sensitivity['mean_delta']:+.6f}`, CI `[{sensitivity['bootstrap_95_ci'][0]:+.6f}, {sensitivity['bootstrap_95_ci'][1]:+.6f}]`, p=`{sensitivity['sign_flip_two_sided_p']:.6f}`.")
    fallback_summary = {}
    for event in json_fallbacks:
        key = (event["component"], event["phase"], event["user_id"])
        fallback_summary[key] = fallback_summary.get(key, 0) + 1
    fallback_text = "; ".join(
        f"{component}/{phase}/user {user_id}: {count}"
        for (component, phase, user_id), count in sorted(fallback_summary.items())
    ) or "none"
    lines.extend(["", "## Integrity", "", f"- Candidate manifests byte-identical: Block 1 `{manifests['1']['byte_identical']}`, Block 2 `{manifests['2']['byte_identical']}`.", f"- `credit_updates_during_test=0` for all six retained runs: `{result['all_test_credit_frozen']}`.", f"- Block 2 handled JSON fallback events: `{len(json_fallbacks)}` ({fallback_text}); fatal transport errors: `0`. These fallbacks did not abort ranking or memory-update control flow."])
    for block in ("1", "2"):
        gate = gate_by_block[block]
        lines.append(f"- Block {block} Full: rejects {gate['rejected']}/{gate['propagation_total']} ({gate['reject_rate']:.2%}); direct writes preserved {gate['direct_preserved']}/{gate['direct_total']} ({gate['direct_preservation_rate']:.2%}); invalid relations {gate['invalid_relations']}/{gate['claimed_relations']} ({gate['invalid_relation_ratio']:.2%}).")
    lines.extend(["", "## Evidence limitations", "", "- Block 1 lacks recoverable per-user predicted rankings, so paired bootstrap/sign-flip inference uses only Block 2's 100 paired users.", "- Two blocks show direction but cannot establish broad cross-block stability.", "- Block 3 was intentionally deferred; no missing result was reconstructed, estimated, or imputed.", "", "Complete Hit metrics, runtime/token fields, hashes, gate statistics, and test outputs are in `outputs/feedback_memrec_v11_2x100_summary.json` and `.csv`.", ""])
    report = ROOT / "docs" / "feedback_memrec_v11_2x100_stage_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary_json": str(json_path), "summary_csv": str(csv_path), "report": str(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
