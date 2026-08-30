#!/usr/bin/env python3
"""Fail-closed acceptance checks for one locked FeedbackMemRec V1.1 run."""

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise AssertionError(f"invalid JSON {path}:{number}: {error}") from error
    return rows


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("variant", choices=("corrected", "read", "full"))
    parser.add_argument("block", type=int)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    out = args.output_dir

    required = (
        "run_metrics.json",
        "resolved_config.json",
        "candidate_manifest.jsonl",
        "memory.jsonl",
        "evidence_credit.json",
        "per_user_metrics.jsonl",
        "warmup_events.jsonl",
    )
    for name in required:
        path = out / name
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty: {path}")

    metrics_doc = load_json(out / "run_metrics.json")
    config = load_json(out / "resolved_config.json")
    require(metrics_doc.get("config") == config, "run_metrics config != resolved_config")
    require(config.get("evaluation_block") == args.block, "wrong evaluation block")
    require(config.get("seed") == args.seed, "wrong seed")
    require(config.get("n_eval_users") == 100, "n_eval_users != 100")
    require(config.get("n_eval_candidates") == 10, "n_eval_candidates != 10")
    require(config.get("warmup") == {"enabled": True, "rounds": 1}, "warmup changed")
    memrec = config.get("memrec", {})
    require(memrec.get("n_facets") == 7, "n_facets != 7")
    require(memrec.get("temperature") == 0.0, "temperature != 0")
    require(memrec.get("max_tokens") == 2400, "max_tokens != 2400")
    require(config.get("feedback_credit", {}).get("freeze_credit") is True, "test credit not frozen")

    per_user = load_jsonl(out / "per_user_metrics.jsonl")
    test_rows = [row for row in per_user if row.get("split") == "test"]
    require(len(test_rows) == 100, f"test per-user rows != 100: {len(test_rows)}")
    require(len({row["user_id"] for row in test_rows}) == 100, "test users are not unique")
    require(all(row.get("ranking_success") is True for row in test_rows), "test ranking success != 100/100")
    require(not any("error" in row for row in test_rows), "per-user test error recorded")

    manifest = load_jsonl(out / "candidate_manifest.jsonl")
    test_manifest = [row for row in manifest if row.get("phase") == "test"]
    require(len(test_manifest) == 100, f"test manifest rows != 100: {len(test_manifest)}")
    require(len({row["user_id"] for row in test_manifest}) == 100, "manifest test users not unique")
    require(all(len(row.get("candidates", [])) == 10 for row in manifest), "candidate count != 10")

    warmup = load_jsonl(out / "warmup_events.jsonl")
    require(len(warmup) == 100, f"warmup event rows != 100: {len(warmup)}")
    require(not any(row.get("status") == "error" for row in warmup), "warmup JSON/model error recorded")

    metrics = metrics_doc.get("test_metrics", {})
    require(metrics.get("n_stage_r_calls") == 100, "Stage-R test calls != 100")
    require(metrics.get("n_stage_rr_calls") == 100, "Stage-RR test calls != 100")
    require(metrics.get("credit_updates_during_test") == 0, "credit updated during test")
    credit = metrics.get("evidence_credit_stats", {})
    attrib = metrics.get("feedback_attribution_stats", {})

    if args.variant == "corrected":
        require(credit.get("total_updates") == 0 and credit.get("n_relations") == 0,
                "Corrected contains credit state")
        zero_fields = (
            "n_attribution_episodes", "n_counterfactual_calls", "n_propagation_accepted",
            "n_propagation_rejected", "n_direct_user_writes", "n_direct_item_writes",
        )
        require(all(attrib.get(key) == 0 for key in zero_fields), "Corrected contains attribution/gate state")
    else:
        require(credit.get("total_updates", 0) > 0, "credit learning did not run in warmup")
        require(attrib.get("n_attribution_episodes", 0) > 0, "attribution did not run in warmup")

    if args.variant == "full":
        log_dir = out / "feedback_memrec_logs"
        propagation = load_jsonl(log_dir / "propagation_events.jsonl")
        direct = load_jsonl(log_dir / "direct_write_events.jsonl")
        rejected = [row for row in propagation if row.get("decision") == "reject"]
        immediate = [row for row in rejected if row.get("reason") == "negative_current_episode_contribution"]
        require(all(row.get("has_current_attribution") is True for row in immediate),
                "immediate reject without current attribution")
        require(all(row.get("raw_episode_delta") is not None and row["raw_episode_delta"] < 0 for row in immediate),
                "immediate reject with nonnegative raw delta")
        require(not any(row.get("has_current_attribution") is True
                        and row.get("raw_episode_delta") is not None
                        and row["raw_episode_delta"] >= 0
                        and row.get("decision") == "reject"
                        for row in propagation), "positive/zero current contribution was rejected")
        require(all(row.get("decision") == "accept"
                    and row.get("reason") == "direct_interaction_not_gated" for row in direct),
                "direct write was gated")
        require(len(direct) == attrib.get("n_direct_user_writes", 0) + attrib.get("n_direct_item_writes", 0),
                "direct-write event/stat mismatch")

    summary = {
        "output_dir": str(out),
        "variant": args.variant,
        "block": args.block,
        "seed": args.seed,
        "test_rankings": len(test_rows),
        "warmup_updated": sum(row.get("status") == "updated" for row in warmup),
        "warmup_nonupdated": sum(row.get("status") != "updated" for row in warmup),
        "credit_updates_during_test": metrics["credit_updates_during_test"],
        "ndcg_at_10": metrics.get("NDCG@10"),
        "wall_time_seconds": metrics.get("wall_time_seconds"),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ACCEPTANCE FAILED: {error}", file=sys.stderr)
        raise
