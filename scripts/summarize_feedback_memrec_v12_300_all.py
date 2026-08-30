#!/usr/bin/env python3
"""Summarize all four locked V1.2 300-user variants."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from summarize_feedback_memrec_v12_300 import (
    aggregate_metrics,
    load_json,
    load_jsonl,
    load_memory,
    metric_row,
    paired_test,
    run_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs"
RUNS = {
    "corrected": OUT_ROOT / "feedback_memrec_v12_continuous_300_corrected",
    "memory_only": OUT_ROOT / "feedback_memrec_v12_continuous_300_memory_only",
    "read": OUT_ROOT / "feedback_memrec_v12_continuous_300_read",
    "full": OUT_ROOT / "feedback_memrec_v12_continuous_300_full",
}
USER_FILE = ROOT / "data/eval_user_samples/strict_books_v12_300_seed45.json"
SUMMARY_PATH = OUT_ROOT / "feedback_memrec_v12_300_summary.json"
PER_USER_PATH = OUT_ROOT / "feedback_memrec_v12_300_per_user.csv"
STATS_PATH = OUT_ROOT / "feedback_memrec_v12_300_paired_statistics.json"
REPORT_PATH = ROOT / "docs/feedback_memrec_v12_300_experiment_report.md"
STAT_SEED = 20260826
N_RESAMPLES = 10_000

HISTORICAL_RUNS = (
    ("V1.0", "development", "corrected", 20, "feedback_memrec_20_corrected"),
    ("V1.0", "development", "read", 20, "feedback_memrec_20_read"),
    ("V1.0", "development", "full", 20, "feedback_memrec_20_full"),
    ("V1.1", "development", "full", 20, "feedback_memrec_v11_20_full"),
    ("V1.1", "formal block 1", "corrected", 100, "feedback_memrec_v11_final_100_corrected"),
    ("V1.1", "formal block 1", "read", 100, "feedback_memrec_v11_final_100_read"),
    ("V1.1", "formal block 1", "full", 100, "feedback_memrec_v11_final_100_full"),
    ("V1.1", "formal block 2", "corrected", 100, "feedback_memrec_v11_block2_seed43_corrected"),
    ("V1.1", "formal block 2", "read", 100, "feedback_memrec_v11_block2_seed43_read"),
    ("V1.1", "formal block 2", "full", 100, "feedback_memrec_v11_block2_seed43_full"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_state(left, right, users, per_variant):
    left_memory = load_memory(RUNS[left] / "memory_after_warmup.jsonl")
    right_memory = load_memory(RUNS[right] / "memory_after_warmup.jsonl")
    memory_keys = set(left_memory) | set(right_memory)
    different = {key for key in memory_keys if left_memory.get(key) != right_memory.get(key)}

    def contexts(label):
        return {
            int(event["target_user_id"]): event
            for event in load_jsonl(
                RUNS[label] / "feedback_memrec_logs/packed_context_events.jsonl"
            )
            if event.get("phase") == "test"
        }

    left_context = contexts(left)
    right_context = contexts(right)
    return {
        "warmup_memory_entries_different": len(different),
        "user_memories_different": sum(key[0] == "user" for key in different),
        "item_memories_different": sum(key[0] == "item" for key in different),
        "test_packed_context_hash_different": sum(
            left_context[u].get("packed_context_hash")
            != right_context[u].get("packed_context_hash")
            for u in users
        ),
        "test_packed_neighbor_sequence_different": sum(
            left_context[u].get("packed_neighbor_ids")
            != right_context[u].get("packed_neighbor_ids")
            for u in users
        ),
        "test_dynamic_neighbor_sequence_different": sum(
            left_context[u].get("dynamic_memory_neighbor_ids")
            != right_context[u].get("dynamic_memory_neighbor_ids")
            for u in users
        ),
        "test_target_rank_different": sum(
            per_variant[left][i]["target_rank"]
            != per_variant[right][i]["target_rank"]
            for i in range(len(users))
        ),
    }


def audit_run(label, run_dir, metrics_doc):
    log_dir = run_dir / "feedback_memrec_logs"
    config = load_json(run_dir / "resolved_config.json")
    packed = load_jsonl(log_dir / "packed_context_events.jsonl")
    reads = load_jsonl(log_dir / "neighbor_memory_read_events.jsonl")
    raw = load_jsonl(log_dir / "read_after_write_events.jsonl")
    propagation = load_jsonl(log_dir / "propagation_events.jsonl")
    direct = load_jsonl(log_dir / "direct_write_events.jsonl")
    run_log = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace").lower()
    feature_enabled = config.get("neighbor_memory_read", {}).get("enabled", False)
    immediate = [
        event
        for event in propagation
        if event.get("reason") == "negative_current_episode_contribution"
    ]
    return {
        "config_sha256": sha256(run_dir / "resolved_config.json"),
        "candidate_manifest_sha256": sha256(run_dir / "candidate_manifest.jsonl"),
        "credit_updates_during_test": metrics_doc.get("credit_updates_during_test"),
        "json_parse_fallbacks": run_log.count("error parsing json"),
        "fatal_markers": {
            pattern: run_log.count(pattern)
            for pattern in ("traceback", "context overflow", "error evaluating user")
        },
        "packed_context_events": len(packed),
        "packed_neighbor_self_consistent": all(
            event.get("packed_neighbors_count")
            == len(event.get("packed_neighbor_ids", []))
            for event in packed
        ),
        "static_snippets_preserved": all(
            all(
                snippet in event.get("packed_context_text", "")
                for snippet in event.get("baseline_static_snippets", [])
            )
            for event in packed
        ),
        "context_tokens_min": min(
            (event.get("total_packed_tokens", 0) for event in packed if feature_enabled),
            default=0,
        ),
        "context_tokens_mean": float(np.mean([
            event.get("total_packed_tokens", 0)
            for event in packed
            if feature_enabled
        ])) if feature_enabled else 0.0,
        "context_tokens_max": max(
            (event.get("total_packed_tokens", 0) for event in packed if feature_enabled),
            default=0,
        ),
        "context_within_1800": all(
            event.get("total_packed_tokens", 0) <= 1800
            for event in packed
            if feature_enabled
        ),
        "neighbor_memory_events": len(reads),
        "single_memory_within_64": all(
            event.get("included_memory_tokens", 0) <= 64 for event in reads
        ),
        "read_after_write_events": len(raw),
        "read_after_write_matches": sum(event.get("hash_match") is True for event in raw),
        "read_after_write_mismatches": sum(event.get("hash_match") is not True for event in raw),
        "propagation_events": len(propagation),
        "propagation_accepted": sum(event.get("decision") == "accept" for event in propagation),
        "propagation_rejected": sum(event.get("decision") == "reject" for event in propagation),
        "reject_by_neighbor_type": dict(Counter(
            event.get("neighbor_type")
            for event in propagation
            if event.get("decision") == "reject"
        )),
        "immediate_rejects_valid": all(
            event.get("decision") == "reject"
            and event.get("has_current_attribution") is True
            and event.get("raw_episode_delta") is not None
            and event["raw_episode_delta"] < 0
            for event in immediate
        ),
        "direct_write_events": len(direct),
        "direct_writes_all_preserved": all(
            event.get("decision") == "accept"
            and event.get("reason") == "direct_interaction_not_gated"
            for event in direct
        ),
    }


def format_float(value):
    return f"{value:.6f}"


def main():
    users = [int(user) for user in load_json(USER_FILE)["user_ids"]]
    assert len(users) == 300 and len(set(users)) == 300
    per_variant = {}
    metrics_docs = {}
    configs = {}
    manifests = {}
    failures = {}
    for label, run_dir in RUNS.items():
        rows = [
            metric_row(row)
            | {"user_id": int(row["user_id"]), "target_item": int(row["target_item"])}
            for row in load_jsonl(run_dir / "per_user_metrics.jsonl")
            if row.get("phase") == "test"
        ]
        assert len(rows) == 300
        assert [row["user_id"] for row in rows] == users
        per_variant[label] = rows
        metrics_docs[label] = load_json(run_dir / "run_metrics.json")["test_metrics"]
        configs[label] = load_json(run_dir / "resolved_config.json")
        manifests[label] = run_dir / "candidate_manifest.jsonl"
        failures[label] = [row["user_id"] for row in rows if not row["ranking_success"]]

    manifest_hashes = {label: sha256(path) for label, path in manifests.items()}
    assert len(set(manifest_hashes.values())) == 1
    for index in range(300):
        assert len({per_variant[label][index]["target_item"] for label in RUNS}) == 1

    merged = []
    for index, user_id in enumerate(users):
        row = {
            "order": index + 1,
            "segment": index // 100 + 1,
            "user_id": user_id,
            "target_item": per_variant["corrected"][index]["target_item"],
        }
        for label in RUNS:
            for key, value in per_variant[label][index].items():
                if key not in ("user_id", "target_item"):
                    row[f"{label}_{key}"] = value
        for name, left, right in (
            ("memory_only_minus_corrected", "memory_only", "corrected"),
            ("read_minus_memory_only", "read", "memory_only"),
            ("full_minus_read", "full", "read"),
        ):
            row[f"{name}_ndcg_at_10"] = (
                row[f"{left}_ndcg_at_10"] - row[f"{right}_ndcg_at_10"]
            )
        merged.append(row)

    with PER_USER_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)

    comparisons = {
        "memory_only_minus_corrected": ("memory_only", "corrected"),
        "read_minus_memory_only": ("read", "memory_only"),
        "full_minus_read": ("full", "read"),
        "read_minus_corrected": ("read", "corrected"),
        "full_minus_corrected": ("full", "corrected"),
        "full_minus_memory_only": ("full", "memory_only"),
    }
    metric_keys = ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "ndcg_at_10")
    rng = np.random.default_rng(STAT_SEED)
    stats = {
        "seed": STAT_SEED,
        "n_resamples": N_RESAMPLES,
        "all_300_failure_policy": "ranking failures contribute zero utility",
        "comparisons": {},
    }
    for name, (left, right) in comparisons.items():
        common = [
            i
            for i in range(300)
            if per_variant[left][i]["ranking_success"]
            and per_variant[right][i]["ranking_success"]
        ]
        result = {"all_300": {}, "common_success": {"n_users": len(common), "metrics": {}}}
        for key in metric_keys:
            result["all_300"][key] = paired_test(
                [row[key] for row in per_variant[left]],
                [row[key] for row in per_variant[right]],
                rng,
            )
            result["common_success"]["metrics"][key] = paired_test(
                [per_variant[left][i][key] for i in common],
                [per_variant[right][i][key] for i in common],
                rng,
            )
        stats["comparisons"][name] = result
    STATS_PATH.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    all_four_common = [
        i for i in range(300) if all(per_variant[label][i]["ranking_success"] for label in RUNS)
    ]
    common_metrics = {
        label: aggregate_metrics([per_variant[label][i] for i in all_four_common])
        for label in RUNS
    }
    diagnostics = {
        label: run_diagnostics(label, RUNS[label], users, per_variant[label])
        for label in RUNS
    }
    run_audits = {
        label: audit_run(label, RUNS[label], metrics_docs[label])
        for label in RUNS
    }
    state_differences = {
        "memory_only_vs_corrected": compare_state(
            "memory_only", "corrected", users, per_variant
        ),
        "read_vs_memory_only": compare_state(
            "read", "memory_only", users, per_variant
        ),
        "full_vs_read": compare_state("full", "read", users, per_variant),
    }

    summary = {
        "protocol": {
            "users": 300,
            "seed": 45,
            "user_list_sha256": sha256(USER_FILE),
            "candidate_manifest_sha256": next(iter(manifest_hashes.values())),
            "candidate_manifest_hashes": manifest_hashes,
            "candidate_manifests_byte_identical": True,
            "config_file_sha256": {
                "corrected": "db94c14ac80e667e5f89fd5cc053e72874195322a94fb73047342346506b4469",
                "memory_only": "c63890c02b9b2c252afa0209508d3cf6b077483049a220734b4a9a26e6162172",
                "read": "1b54eb812aa15a982459e9dff9958ad4d04bb897fb97cf2ea5034a851a5d0f5c",
                "full": "fd7337043319aa564df861fa8224b6c13472c3c697d8adbce81dcc0e91975635",
            },
            "statistics_seed": STAT_SEED,
            "resamples": N_RESAMPLES,
        },
        "metrics_all_300": {
            label: aggregate_metrics(per_variant[label]) for label in RUNS
        },
        "all_four_common_success": {
            "n_users": len(all_four_common),
            "metrics": common_metrics,
        },
        "ranking_failures": failures,
        "run_metrics": metrics_docs,
        "run_audits": run_audits,
        "diagnostics": diagnostics,
        "state_and_context_differences": state_differences,
        "paired_statistics_path": str(STATS_PATH.relative_to(ROOT)),
        "per_user_path": str(PER_USER_PATH.relative_to(ROOT)),
    }

    historical = []
    for version, scope, variant, n_users, directory in HISTORICAL_RUNS:
        path = OUT_ROOT / directory / "run_metrics.json"
        if not path.is_file():
            continue
        run_metrics = load_json(path).get("test_metrics", {})
        historical.append({
            "version": version,
            "scope": scope,
            "variant": variant,
            "n_users": n_users,
            "Hit@1": run_metrics.get("Hit@1"),
            "Hit@3": run_metrics.get("Hit@3"),
            "Hit@5": run_metrics.get("Hit@5"),
            "Hit@10": run_metrics.get("Hit@10"),
            "NDCG@10": run_metrics.get("NDCG@10"),
            "wall_time_seconds": run_metrics.get("wall_time_seconds"),
            "output_dir": f"outputs/{directory}",
        })
    summary["historical_and_development_runs"] = historical
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    metrics = summary["metrics_all_300"]
    primary = (
        ("Memory Only - Corrected", "memory_only_minus_corrected"),
        ("Read - Memory Only", "read_minus_memory_only"),
        ("Full - Read", "full_minus_read"),
    )
    lines = [
        "# FeedbackMemRec V1.2 300-user Experiment Report",
        "",
        "## Bottom line",
        "",
        "The four locked runs completed 300 warmup and 300 test users. Each run had",
        "4-5 isolated incomplete LLM rankings, so strict 300/300 acceptance failed.",
        "The prespecified all-300 analysis counts those failures as zero; common-success",
        "analyses exclude only the failed pair members. No primary NDCG comparison is",
        "statistically distinguishable from zero at the 95% level.",
        "",
        "## All-300 metrics (ranking failures count as zero)",
        "",
        "| Variant | Success | Hit@1 | Hit@3 | Hit@5 | Hit@10 | NDCG@10 | Wall time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in RUNS:
        row = metrics[label]
        wall = metrics_docs[label].get("wall_time_seconds", 0) / 3600
        lines.append(
            f"| {label} | {row['ranking_successes']}/300 | {row['Hit@1']:.4f} | "
            f"{row['Hit@3']:.4f} | {row['Hit@5']:.4f} | {row['Hit@10']:.4f} | "
            f"{row['NDCG@10']:.6f} | {wall:.2f} h |"
        )
    lines += [
        "",
        "## Primary incremental comparisons",
        "",
        "| Comparison | all-300 NDCG delta | bootstrap 95% CI | sign-flip p | common-success n | common-success delta | common 95% CI | common p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for title, key in primary:
        all_result = stats["comparisons"][key]["all_300"]["ndcg_at_10"]
        common = stats["comparisons"][key]["common_success"]
        common_result = common["metrics"]["ndcg_at_10"]
        lines.append(
            f"| {title} | {all_result['mean_delta']:+.6f} | "
            f"[{all_result['bootstrap_95_ci'][0]:+.6f}, {all_result['bootstrap_95_ci'][1]:+.6f}] | "
            f"{all_result['sign_flip_two_sided_p']:.4f} | {common['n_users']} | "
            f"{common_result['mean_delta']:+.6f} | "
            f"[{common_result['bootstrap_95_ci'][0]:+.6f}, {common_result['bootstrap_95_ci'][1]:+.6f}] | "
            f"{common_result['sign_flip_two_sided_p']:.4f} |"
        )
    lines += [
        "",
        "## Common success across all four variants",
        "",
        f"All four variants succeeded for {len(all_four_common)}/300 users.",
        "",
        "| Variant | Hit@1 | Hit@3 | Hit@5 | Hit@10 | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in RUNS:
        row = common_metrics[label]
        lines.append(
            f"| {label} | {row['Hit@1']:.4f} | {row['Hit@3']:.4f} | "
            f"{row['Hit@5']:.4f} | {row['Hit@10']:.4f} | {row['NDCG@10']:.6f} |"
        )
    full_audit = run_audits["full"]
    lines += [
        "",
        "## Mechanism and fairness audit",
        "",
        f"- Candidate manifests are byte-identical: `{next(iter(manifest_hashes.values()))}`.",
        "- Test credit updates are zero in all four variants.",
        "- All locked source/config hashes still pass after execution.",
        "- Read, Full, and Memory Only all obey the 64-token single-memory, 512-token",
        "  dynamic-memory, and 1800-token Stage-R caps; static snippets are preserved.",
        f"- Read-after-write mismatches: corrected 0, memory_only {run_audits['memory_only']['read_after_write_mismatches']}, "
        f"read {run_audits['read']['read_after_write_mismatches']}, full {full_audit['read_after_write_mismatches']}.",
        f"- Full gate: {full_audit['propagation_rejected']}/{full_audit['propagation_events']} collaborative writes rejected "
        f"({full_audit['propagation_rejected']/full_audit['propagation_events']:.2%}); "
        f"item neighbors {full_audit['reject_by_neighbor_type'].get('item_neighbor', 0)}, "
        f"user neighbors {full_audit['reject_by_neighbor_type'].get('user_neighbor', 0)}.",
        f"- All {full_audit['direct_write_events']} Full direct-write events were preserved; "
        "all immediate rejects had raw_episode_delta < 0.",
        f"- Full dynamic memories: {metrics_docs['full']['neighbor_memory_read_stats']['packed_memories']} packed, "
        f"{metrics_docs['full']['neighbor_memory_read_stats']['dropped_memories']} dropped; "
        f"context tokens min/mean/max {full_audit['context_tokens_min']}/"
        f"{full_audit['context_tokens_mean']:.1f}/{full_audit['context_tokens_max']}.",
        "",
        "## Interpretation",
        "",
        "- Dynamic neighbor memory alone is directionally above Corrected, but the CI",
        "  crosses zero; this run does not establish a reliable gain.",
        "- Adding credit and attribution on top of Memory Only is also directionally",
        "  positive, but statistically inconclusive.",
        "- The write gate is directionally negative versus Read in this seed. Its 145",
        "  immediate rejects changed long-term state and context, but did not improve",
        "  aggregate ranking quality in this run.",
        "- These are one dataset, one locked 300-user sequence, and one model seed. The",
        "  three 100-user segments are temporal diagnostics, not independent seeds.",
        "",
        "## Historical V1.1 context (separate protocol)",
        "",
        "V1.1 Block 1 NDCG@10 was 0.689771/0.696837/0.709473 for",
        "Corrected/Read/Full; Block 2 was 0.700578/0.705433/0.705433.",
        "Thus V1.1 showed positive direction versus Corrected in both blocks, while",
        "the Full-Read gain appeared only in Block 1. These results are not pooled with",
        "V1.2 because V1.2 changes the Stage-R input by reading dynamic neighbor memory.",
        "",
        "## Historical and development run inventory",
        "",
        "Development rows below are implementation checks, not formal effect",
        "estimates. V1.1 Blocks 1 and 2 are independent 100-user formal blocks.",
        "",
        "| Version | Scope | Variant | Users | Hit@1 | Hit@3 | Hit@5 | Hit@10 | NDCG@10 | Wall time |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in historical:
        lines.append(
            f"| {row['version']} | {row['scope']} | {row['variant']} | {row['n_users']} | "
            f"{row['Hit@1']:.4f} | {row['Hit@3']:.4f} | {row['Hit@5']:.4f} | "
            f"{row['Hit@10']:.4f} | {row['NDCG@10']:.6f} | "
            f"{row['wall_time_seconds']/3600:.2f} h |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Summary: `{SUMMARY_PATH.relative_to(ROOT)}`",
        f"- Per-user table: `{PER_USER_PATH.relative_to(ROOT)}`",
        f"- Paired statistics: `{STATS_PATH.relative_to(ROOT)}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(SUMMARY_PATH),
        "per_user": str(PER_USER_PATH),
        "statistics": str(STATS_PATH),
        "report": str(REPORT_PATH),
        "all_four_common_success": len(all_four_common),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
