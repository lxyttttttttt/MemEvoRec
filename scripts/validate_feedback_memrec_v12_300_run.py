#!/usr/bin/env python3
"""Fail-closed acceptance checks for one locked V1.2 300-user run."""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path):
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


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("variant", choices=("corrected", "read", "full"))
    args = parser.parse_args()
    out = args.output_dir
    log_dir = out / "feedback_memrec_logs"

    required = (
        "run_metrics.json",
        "resolved_config.json",
        "candidate_manifest.jsonl",
        "memory_after_warmup.jsonl",
        "memory.jsonl",
        "evidence_credit.json",
        "per_user_metrics.jsonl",
        "warmup_events.jsonl",
        "run.log",
        "feedback_memrec_logs/packed_context_events.jsonl",
    )
    for name in required:
        path = out / name
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty: {path}")

    config = load_json(out / "resolved_config.json")
    metrics_doc = load_json(out / "run_metrics.json")
    require(metrics_doc.get("config") == config, "run_metrics config != resolved_config")
    require(config.get("evaluation_block") == "v12_continuous_300", "wrong block")
    require(config.get("seed") == 45, "seed != 45")
    require(config.get("n_eval_users") == 300, "n_eval_users != 300")
    require(config.get("n_eval_candidates") == 10, "candidate count config != 10")
    require(config.get("warmup") == {"enabled": True, "rounds": 1}, "warmup changed")
    require(config.get("persist_memory_after_warmup") is True, "warmup snapshot disabled")
    require(config.get("eval_user_list") == "data/eval_user_samples/strict_books_v12_300_seed45.json", "wrong user list")
    memrec = config.get("memrec", {})
    require(memrec.get("k") == 16, "neighbor k changed")
    require(memrec.get("tau_tokens") == 1800, "Stage-R budget changed")
    require(memrec.get("n_facets") == 7, "facet count changed")
    require(memrec.get("temperature") == 0.0, "temperature changed")
    require(config.get("feedback_credit", {}).get("freeze_credit") is True, "test credit not frozen")

    expected = {
        "corrected": (False, False, False, False, False),
        "read": (True, True, True, False, True),
        "full": (True, True, True, True, True),
    }[args.variant]
    actual = (
        config.get("neighbor_memory_read", {}).get("enabled", False),
        config.get("feedback_credit", {}).get("enabled", False),
        config.get("read_credit", {}).get("enabled", False),
        config.get("write_gate", {}).get("enabled", False),
        config.get("attribution", {}).get("enabled", False),
    )
    require(actual == expected, f"wrong feature flags: {actual} != {expected}")
    nmr_cfg = config.get("neighbor_memory_read", {})
    require(nmr_cfg.get("per_memory_tokens") == 64, "per-memory budget changed")
    require(nmr_cfg.get("total_memory_tokens") == 512, "dynamic budget changed")
    require(nmr_cfg.get("preserve_static_neighbors") is True, "static neighbors not preserved")
    if args.variant == "full":
        require(config.get("write_gate", {}).get("use_current_episode_delta") is True, "immediate gate disabled")

    users = load_json(Path(config["eval_user_list"]))["user_ids"]
    require(len(users) == 300 and len(set(users)) == 300, "locked user list invalid")

    per_user = load_jsonl(out / "per_user_metrics.jsonl")
    test_rows = [row for row in per_user if row.get("phase") == "test"]
    require(len(test_rows) == 300, f"test rows != 300: {len(test_rows)}")
    require([row["user_id"] for row in test_rows] == users, "test user order changed")
    require(all(row.get("ranking_success") is True for row in test_rows), "test ranking failure")
    require(not any("error" in row for row in test_rows), "test error recorded")

    warmup = load_jsonl(out / "warmup_events.jsonl")
    require(len(warmup) == 300, f"warmup rows != 300: {len(warmup)}")
    require([row["user_id"] for row in warmup] == users, "warmup user order changed")
    require(not any(row.get("status") == "error" for row in warmup), "warmup error recorded")

    manifest = load_jsonl(out / "candidate_manifest.jsonl")
    warm_manifest = [row for row in manifest if row.get("phase") == "warmup-0"]
    test_manifest = [row for row in manifest if row.get("phase") == "test"]
    require(len(warm_manifest) == 300 and len(test_manifest) == 300, "manifest phase count changed")
    require([row["user_id"] for row in warm_manifest] == users, "warmup manifest order changed")
    require([row["user_id"] for row in test_manifest] == users, "test manifest order changed")
    require(all(len(row.get("candidates", [])) == 10 for row in manifest), "candidate length != 10")

    packed = load_jsonl(log_dir / "packed_context_events.jsonl")
    require(len(packed) == 600, f"packed context episodes != 600: {len(packed)}")
    for event in packed:
        require(event.get("packed_neighbors_count") == len(event.get("packed_neighbor_ids", [])), "packed count/id mismatch")
        text = event.get("packed_context_text", "")
        for snippet in event.get("baseline_static_snippets", []):
            require(snippet in text, "baseline static snippet missing from packed context")

    metrics = metrics_doc.get("test_metrics", {})
    require(metrics.get("n_stage_r_calls") == 300, "Stage-R test calls != 300")
    require(metrics.get("n_stage_rr_calls") == 300, "Stage-RR test calls != 300")
    require(metrics.get("credit_updates_during_test") == 0, "credit updated during test")
    credit = metrics.get("evidence_credit_stats", {})
    attrib = metrics.get("feedback_attribution_stats", {})
    nmr_stats = metrics.get("neighbor_memory_read_stats", {})

    if args.variant == "corrected":
        require(credit.get("total_updates") == 0 and credit.get("n_relations") == 0, "Corrected contains credit state")
        require(nmr_stats.get("episodes") == 0 and nmr_stats.get("packed_memories") == 0, "Corrected contains dynamic-read state")
        require(not (log_dir / "neighbor_memory_read_events.jsonl").exists(), "Corrected emitted neighbor-memory events")
        require(not (log_dir / "read_after_write_events.jsonl").exists(), "Corrected emitted read-after-write events")
    else:
        require(credit.get("total_updates", 0) > 0, "credit learning did not run")
        require(attrib.get("n_attribution_episodes", 0) > 0, "attribution did not run")
        memory_reads = load_jsonl(log_dir / "neighbor_memory_read_events.jsonl")
        require(memory_reads, "no neighbor memory read events")
        require(all(row.get("included_memory_tokens", 0) <= 64 for row in memory_reads), "single dynamic memory >64")
        require(all(event.get("dynamic_memory_tokens", 0) <= 512 for event in packed), "episode dynamic memory >512")
        require(all(event.get("total_packed_tokens", 0) <= 1800 for event in packed), "packed context >1800")
        raw_path = log_dir / "read_after_write_events.jsonl"
        raw = load_jsonl(raw_path) if raw_path.exists() else []
        require(not any(not row.get("hash_match", False) for row in raw), "read-after-write hash mismatch")

    if args.variant == "full":
        propagation = load_jsonl(log_dir / "propagation_events.jsonl")
        direct = load_jsonl(log_dir / "direct_write_events.jsonl")
        immediate = [row for row in propagation if row.get("reason") == "negative_current_episode_contribution"]
        require(all(row.get("decision") == "reject" and row.get("has_current_attribution") is True and row.get("raw_episode_delta") is not None and row["raw_episode_delta"] < 0 for row in immediate), "invalid immediate reject")
        require(all(row.get("decision") == "accept" and row.get("reason") == "direct_interaction_not_gated" for row in direct), "direct write was gated")
        require(len(direct) == attrib.get("n_direct_user_writes", 0) + attrib.get("n_direct_item_writes", 0), "direct event/stat mismatch")

    # Frozen model-side JSON fallbacks are recorded but are not themselves a
    # failed run when the user still receives a complete ranking. Per-user
    # ranking/warmup failures are checked structurally above.
    bad_patterns = ("traceback", "context overflow", "error evaluating user")
    run_log = (out / "run.log").read_text(encoding="utf-8", errors="replace").lower()
    require(not any(pattern in run_log for pattern in bad_patterns), "run log contains model/JSON/runtime error")
    json_fallback_count = run_log.count("error parsing json")

    print(json.dumps({
        "variant": args.variant,
        "output_dir": str(out),
        "test_rankings": len(test_rows),
        "warmup_rows": len(warmup),
        "credit_updates_during_test": metrics["credit_updates_during_test"],
        "json_fallback_count": json_fallback_count,
        "ndcg_at_10": metrics.get("NDCG@10"),
        "wall_time_seconds": metrics.get("wall_time_seconds"),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ACCEPTANCE FAILED: {error}", file=__import__("sys").stderr)
        raise
