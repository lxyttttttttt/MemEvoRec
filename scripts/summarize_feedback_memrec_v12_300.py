#!/usr/bin/env python3
"""Aggregate the locked V1.2 300-user experiment after all runs pass."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs"
RUNS = {
    "corrected": OUT_ROOT / "feedback_memrec_v12_continuous_300_corrected",
    "read": OUT_ROOT / "feedback_memrec_v12_continuous_300_read",
    "full": OUT_ROOT / "feedback_memrec_v12_continuous_300_full",
}
USER_FILE = ROOT / "data/eval_user_samples/strict_books_v12_300_seed45.json"
SUMMARY_PATH = OUT_ROOT / "feedback_memrec_v12_300_summary.json"
PER_USER_PATH = OUT_ROOT / "feedback_memrec_v12_300_per_user.csv"
STATS_PATH = OUT_ROOT / "feedback_memrec_v12_300_paired_statistics.json"
STAT_SEED = 20260826
N_RESAMPLES = 10_000


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssertionError(f"invalid JSON {path}:{number}: {error}") from error
    return rows


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return float(np.mean(values)) if values else 0.0


def metric_row(row):
    return {
        "target_rank": int(row["target_rank"]),
        "hit_at_1": int(row["hit_at_1"]),
        "hit_at_3": int(row["hit_at_3"]),
        "hit_at_5": int(row["hit_at_5"]),
        "hit_at_10": int(row["hit_at_10"]),
        "ndcg_at_10": float(row["ndcg_at_10"]),
        "ranking_success": bool(row["ranking_success"]),
    }


def aggregate_metrics(rows):
    return {
        "n_users": len(rows),
        "ranking_successes": sum(row["ranking_success"] for row in rows),
        "mean_target_rank_zero_based": mean([row["target_rank"] for row in rows]),
        "Hit@1": mean([row["hit_at_1"] for row in rows]),
        "Hit@3": mean([row["hit_at_3"] for row in rows]),
        "Hit@5": mean([row["hit_at_5"] for row in rows]),
        "Hit@10": mean([row["hit_at_10"] for row in rows]),
        "NDCG@10": mean([row["ndcg_at_10"] for row in rows]),
    }


def paired_test(a, b, rng):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    observed = float(diffs.mean())
    n = len(diffs)
    samples = rng.integers(0, n, size=(N_RESAMPLES, n))
    bootstrap = diffs[samples].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_RESAMPLES, n))
    permuted = (signs * diffs).mean(axis=1)
    p_value = float((np.count_nonzero(np.abs(permuted) >= abs(observed)) + 1) / (N_RESAMPLES + 1))
    return {
        "n_pairs": n,
        "mean_delta": observed,
        "bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "sign_flip_two_sided_p": p_value,
        "n_resamples": N_RESAMPLES,
    }


def parse_episode_user(episode_id):
    match = re.search(r"-u(\d+)(?:-|$)", str(episode_id))
    return int(match.group(1)) if match else None


def phase_of_episode(episode_id):
    return "warmup" if str(episode_id).startswith("warmup-") else "test"


def segment_of_user(user_id, user_position):
    index = user_position.get(int(user_id))
    if index is None:
        return None
    return f"{(index // 100) * 100 + 1}-{(index // 100 + 1) * 100}"


def load_memory(path):
    return {
        (row["type"], int(row["id"])): row["memory"]
        for row in load_jsonl(path)
    }


def run_diagnostics(label, run_dir, users, per_user):
    log_dir = run_dir / "feedback_memrec_logs"
    packed = load_jsonl(log_dir / "packed_context_events.jsonl")
    reads = load_jsonl(log_dir / "neighbor_memory_read_events.jsonl")
    raw = load_jsonl(log_dir / "read_after_write_events.jsonl")
    facets = load_jsonl(log_dir / "facet_attribution_events.jsonl")
    propagation = load_jsonl(log_dir / "propagation_events.jsonl")
    direct = load_jsonl(log_dir / "direct_write_events.jsonl")
    position = {user_id: index for index, user_id in enumerate(users)}
    segments = {}
    for start in (0, 100, 200):
        key = f"{start + 1}-{start + 100}"
        segment_users = set(users[start:start + 100])
        segment_rows = [row for row in per_user if row["user_id"] in segment_users]
        segment_packed = [event for event in packed if event.get("target_user_id") in segment_users]
        segment_reads = [event for event in reads if event.get("target_user_id") in segment_users]
        segment_raw = [event for event in raw if event.get("target_user_id") in segment_users]
        segment_prop = [event for event in propagation if event.get("target_user_id") in segment_users]
        segment_facets = [event for event in facets if event.get("target_user_id") in segment_users]
        relations = set()
        for event in segment_facets:
            for relation in event.get("validated_supporting_relations", []):
                relations.add((relation["target_user_id"], relation["evidence_type"], relation["evidence_id"]))
        rejects = [event for event in segment_prop if event.get("decision") == "reject"]
        segments[key] = {
            "metrics": aggregate_metrics(segment_rows),
            "validated_credit_relations": len(relations),
            "facet_attribution_events": len(segment_facets),
            "propagation_candidates": len(segment_prop),
            "rejects": len(rejects),
            "reject_rate": len(rejects) / len(segment_prop) if segment_prop else 0.0,
            "dynamic_memory_candidates": len(segment_reads),
            "dynamic_memory_packed": sum(event.get("entered_packed_context", False) for event in segment_reads),
            "dynamic_memory_tokens": sum(event.get("included_memory_tokens", 0) for event in segment_reads),
            "read_after_write_matches": sum(event.get("hash_match", False) for event in segment_raw),
            "read_after_write_mismatches": sum(not event.get("hash_match", False) for event in segment_raw),
            "packed_context_tokens_mean": mean([event.get("total_packed_tokens", 0) for event in segment_packed if event.get("feature_enabled")]),
        }

    reject_events = [event for event in propagation if event.get("decision") == "reject"]
    return {
        "segments": segments,
        "packed_context": {
            "episodes": len(packed),
            "tokens_min": min((event.get("total_packed_tokens", 0) for event in packed if event.get("feature_enabled")), default=0),
            "tokens_mean": mean([event.get("total_packed_tokens", 0) for event in packed if event.get("feature_enabled")]),
            "tokens_max": max((event.get("total_packed_tokens", 0) for event in packed if event.get("feature_enabled")), default=0),
        },
        "neighbor_memory": {
            "candidate_events": len(reads),
            "packed": sum(event.get("entered_packed_context", False) for event in reads),
            "dropped": sum(not event.get("entered_packed_context", False) for event in reads),
            "drop_reasons": dict(Counter(event.get("drop_reason") for event in reads if not event.get("entered_packed_context", False))),
        },
        "read_after_write": {
            "events": len(raw),
            "matches": sum(event.get("hash_match", False) for event in raw),
            "mismatches": sum(not event.get("hash_match", False) for event in raw),
        },
        "gate": {
            "propagation_candidates": len(propagation),
            "accepted": sum(event.get("decision") == "accept" for event in propagation),
            "rejected": len(reject_events),
            "reject_rate": len(reject_events) / len(propagation) if propagation else 0.0,
            "reject_by_type": dict(Counter(event.get("neighbor_type") for event in reject_events)),
            "reject_by_reason": dict(Counter(event.get("reason") for event in reject_events)),
            "direct_write_events": len(direct),
            "direct_all_preserved": all(event.get("decision") == "accept" and event.get("reason") == "direct_interaction_not_gated" for event in direct),
        },
    }


def main():
    users = load_json(USER_FILE)["user_ids"]
    per_variant = {}
    metrics_docs = {}
    manifests = {}
    for label, run_dir in RUNS.items():
        rows = [metric_row(row) | {"user_id": int(row["user_id"]), "target_item": int(row["target_item"])} for row in load_jsonl(run_dir / "per_user_metrics.jsonl") if row.get("phase") == "test"]
        assert [row["user_id"] for row in rows] == users
        per_variant[label] = rows
        metrics_docs[label] = load_json(run_dir / "run_metrics.json")["test_metrics"]
        manifests[label] = run_dir / "candidate_manifest.jsonl"

    assert manifests["corrected"].read_bytes() == manifests["read"].read_bytes() == manifests["full"].read_bytes()

    merged = []
    for index, user_id in enumerate(users):
        row = {"order": index + 1, "segment": f"{index // 100 + 1}", "user_id": user_id, "target_item": per_variant["corrected"][index]["target_item"]}
        for label in ("corrected", "read", "full"):
            for key, value in per_variant[label][index].items():
                if key not in ("user_id", "target_item"):
                    row[f"{label}_{key}"] = value
        row["full_minus_read_ndcg_at_10"] = row["full_ndcg_at_10"] - row["read_ndcg_at_10"]
        row["full_minus_corrected_ndcg_at_10"] = row["full_ndcg_at_10"] - row["corrected_ndcg_at_10"]
        row["read_minus_corrected_ndcg_at_10"] = row["read_ndcg_at_10"] - row["corrected_ndcg_at_10"]
        merged.append(row)

    with PER_USER_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)

    comparisons = {
        "full_minus_read": ("full", "read"),
        "full_minus_corrected": ("full", "corrected"),
        "read_minus_corrected": ("read", "corrected"),
    }
    rng = np.random.default_rng(STAT_SEED)
    stats = {"seed": STAT_SEED, "n_resamples": N_RESAMPLES, "comparisons": {}}
    keys = ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "ndcg_at_10")
    for name, (left, right) in comparisons.items():
        stats["comparisons"][name] = {}
        for key in keys:
            stats["comparisons"][name][key] = paired_test(
                [row[key] for row in per_variant[left]],
                [row[key] for row in per_variant[right]],
                rng,
            )
        common = [i for i in range(300) if per_variant[left][i]["ranking_success"] and per_variant[right][i]["ranking_success"]]
        stats["comparisons"][name]["common_success"] = {
            "n_users": len(common),
            "ndcg_at_10": paired_test(
                [per_variant[left][i]["ndcg_at_10"] for i in common],
                [per_variant[right][i]["ndcg_at_10"] for i in common],
                rng,
            ) if common else None,
        }
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    diagnostics = {
        label: run_diagnostics(label, RUNS[label], users, per_variant[label])
        for label in RUNS
    }

    read_memory = load_memory(RUNS["read"] / "memory_after_warmup.jsonl")
    full_memory = load_memory(RUNS["full"] / "memory_after_warmup.jsonl")
    memory_keys = set(read_memory) | set(full_memory)
    different_memory_keys = {key for key in memory_keys if read_memory.get(key) != full_memory.get(key)}
    read_test_contexts = {event["target_user_id"]: event for event in load_jsonl(RUNS["read"] / "feedback_memrec_logs/packed_context_events.jsonl") if event.get("phase") == "test"}
    full_test_contexts = {event["target_user_id"]: event for event in load_jsonl(RUNS["full"] / "feedback_memrec_logs/packed_context_events.jsonl") if event.get("phase") == "test"}
    context_differences = {
        "warmup_memory_entries_different": len(different_memory_keys),
        "user_memories_different": sum(key[0] == "user" for key in different_memory_keys),
        "item_memories_different": sum(key[0] == "item" for key in different_memory_keys),
        "test_packed_context_hash_different": sum(read_test_contexts[u]["packed_context_hash"] != full_test_contexts[u]["packed_context_hash"] for u in users),
        "test_packed_neighbor_sequence_different": sum(read_test_contexts[u]["packed_neighbor_ids"] != full_test_contexts[u]["packed_neighbor_ids"] for u in users),
        "test_dynamic_neighbor_sequence_different": sum(read_test_contexts[u]["dynamic_memory_neighbor_ids"] != full_test_contexts[u]["dynamic_memory_neighbor_ids"] for u in users),
        "test_target_rank_different": sum(per_variant["read"][i]["target_rank"] != per_variant["full"][i]["target_rank"] for i in range(300)),
    }

    full_prop = load_jsonl(RUNS["full"] / "feedback_memrec_logs/propagation_events.jsonl")
    full_reads = load_jsonl(RUNS["full"] / "feedback_memrec_logs/neighbor_memory_read_events.jsonl")
    user_position = {u: i for i, u in enumerate(users)}
    rejected_then_read = 0
    for rejection in [event for event in full_prop if event.get("decision") == "reject"]:
        source_user = rejection.get("target_user_id")
        source_order = user_position.get(source_user, -1) + (0 if phase_of_episode(rejection.get("episode_id")) == "warmup" else 300)
        key = (rejection.get("neighbor_type"), rejection.get("neighbor_id"))
        if any(
            (event.get("neighbor_type"), event.get("neighbor_id")) == key
            and user_position.get(event.get("target_user_id"), -1) + (0 if event.get("phase") == "warmup" else 300) > source_order
            and event.get("entered_packed_context")
            for event in full_reads
        ):
            rejected_then_read += 1
    diagnostics["full"]["gate"]["rejected_events_with_later_read"] = rejected_then_read

    summary = {
        "protocol": {
            "users": 300,
            "user_list_sha256": sha256(USER_FILE),
            "candidate_manifest_sha256": sha256(manifests["corrected"]),
            "candidate_manifests_byte_identical": True,
            "statistics_seed": STAT_SEED,
            "resamples": N_RESAMPLES,
        },
        "metrics": {label: aggregate_metrics(per_variant[label]) for label in RUNS},
        "run_metrics": metrics_docs,
        "paired_statistics_path": str(STATS_PATH.relative_to(ROOT)),
        "per_user_path": str(PER_USER_PATH.relative_to(ROOT)),
        "diagnostics": diagnostics,
        "read_full_differences": context_differences,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(SUMMARY_PATH),
        "per_user": str(PER_USER_PATH),
        "statistics": str(STATS_PATH),
        "manifest_sha256": summary["protocol"]["candidate_manifest_sha256"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
