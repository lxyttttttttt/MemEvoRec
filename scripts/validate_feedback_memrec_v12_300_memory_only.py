#!/usr/bin/env python3
"""Fail-closed acceptance checks for the locked V1.2 Memory Only ablation."""

import json
from pathlib import Path


OUT = Path("outputs/feedback_memrec_v12_continuous_300_memory_only")
LOG_DIR = OUT / "feedback_memrec_logs"
USER_FILE = Path("data/eval_user_samples/strict_books_v12_300_seed45.json")
REFERENCE_MANIFEST = Path(
    "outputs/feedback_memrec_v12_continuous_300_corrected/candidate_manifest.jsonl"
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
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


def main():
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
        "feedback_memrec_logs/neighbor_memory_read_events.jsonl",
        "feedback_memrec_logs/read_after_write_events.jsonl",
    )
    for name in required:
        path = OUT / name
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty: {path}")

    config = load_json(OUT / "resolved_config.json")
    metrics_doc = load_json(OUT / "run_metrics.json")
    require(metrics_doc.get("config") == config, "run_metrics config != resolved_config")
    require(config.get("evaluation_block") == "v12_continuous_300", "wrong block")
    require(config.get("seed") == 45, "seed != 45")
    require(config.get("n_eval_users") == 300, "n_eval_users != 300")
    require(config.get("n_eval_candidates") == 10, "candidate count != 10")
    require(config.get("warmup") == {"enabled": True, "rounds": 1}, "warmup changed")
    require(config.get("persist_memory_after_warmup") is True, "warmup snapshot disabled")
    require(config.get("eval_user_list") == str(USER_FILE), "wrong user list")

    memrec = config.get("memrec", {})
    require(memrec.get("k") == 16, "neighbor k changed")
    require(memrec.get("tau_tokens") == 1800, "Stage-R budget changed")
    require(memrec.get("n_facets") == 7, "facet count changed")
    require(memrec.get("temperature") == 0.0, "temperature changed")
    require(config.get("feedback_credit", {}).get("enabled") is False, "credit enabled")
    require(config.get("feedback_credit", {}).get("learn_credit") is False, "credit learning enabled")
    require(config.get("feedback_credit", {}).get("freeze_credit") is True, "credit not frozen")
    require(config.get("read_credit", {}).get("enabled") is False, "read credit enabled")
    require(config.get("write_gate", {}).get("enabled") is False, "write gate enabled")
    require(config.get("attribution", {}).get("enabled") is False, "attribution enabled")

    nmr = config.get("neighbor_memory_read", {})
    require(nmr.get("enabled") is True, "neighbor memory disabled")
    require(nmr.get("per_memory_tokens") == 64, "per-memory budget changed")
    require(nmr.get("total_memory_tokens") == 512, "dynamic budget changed")
    require(nmr.get("preserve_static_neighbors") is True, "static neighbors not preserved")
    require(config.get("provider", {}).get("endpoint") == "http://127.0.0.1:8001/v1", "wrong endpoint")
    require(config.get("reranker_provider", {}).get("endpoint") == "http://127.0.0.1:8001/v1", "wrong reranker endpoint")

    users = load_json(USER_FILE)["user_ids"]
    require(len(users) == 300 and len(set(users)) == 300, "locked user list invalid")
    warmup = load_jsonl(OUT / "warmup_events.jsonl")
    test = load_jsonl(OUT / "per_user_metrics.jsonl")
    require(len(warmup) == 300, f"warmup rows != 300: {len(warmup)}")
    require(len(test) == 300, f"test rows != 300: {len(test)}")
    require([row["user_id"] for row in warmup] == users, "warmup user order changed")
    require([row["user_id"] for row in test] == users, "test user order changed")
    require(not any(row.get("status") == "error" for row in warmup), "warmup error recorded")
    require(all(row.get("ranking_success") is True for row in test), "test ranking failure")
    require(not any("error" in row for row in test), "test error recorded")

    manifest_path = OUT / "candidate_manifest.jsonl"
    manifest = load_jsonl(manifest_path)
    require(len(manifest) == 600, f"manifest rows != 600: {len(manifest)}")
    require(manifest_path.read_bytes() == REFERENCE_MANIFEST.read_bytes(), "candidate manifest differs from Corrected")

    packed = load_jsonl(LOG_DIR / "packed_context_events.jsonl")
    require(len(packed) == 600, f"packed context episodes != 600: {len(packed)}")
    for event in packed:
        require(event.get("dynamic_memory_tokens", 0) <= 512, "episode dynamic memory >512")
        require(event.get("total_packed_tokens", 0) <= 1800, "packed context >1800")
        require(event.get("packed_neighbors_count") == len(event.get("packed_neighbor_ids", [])), "packed count/id mismatch")
        text = event.get("packed_context_text", "")
        for snippet in event.get("baseline_static_snippets", []):
            require(snippet in text, "baseline static snippet missing")

    reads = load_jsonl(LOG_DIR / "neighbor_memory_read_events.jsonl")
    require(reads, "no neighbor-memory reads")
    require(all(row.get("included_memory_tokens", 0) <= 64 for row in reads), "single memory >64")
    raw = load_jsonl(LOG_DIR / "read_after_write_events.jsonl")
    require(not any(not row.get("hash_match", False) for row in raw), "read-after-write hash mismatch")

    metrics = metrics_doc.get("test_metrics", {})
    require(metrics.get("n_stage_r_calls") == 300, "Stage-R test calls != 300")
    require(metrics.get("n_stage_rr_calls") == 300, "Stage-RR test calls != 300")
    require(metrics.get("credit_updates_during_test") == 0, "credit updated during test")
    credit = metrics.get("evidence_credit_stats", {})
    attribution = metrics.get("feedback_attribution_stats", {})
    require(credit.get("total_updates") == 0 and credit.get("n_relations") == 0, "credit state is not empty")
    require(attribution.get("n_attribution_episodes") == 0, "attribution ran")

    run_log = (OUT / "run.log").read_text(encoding="utf-8", errors="replace").lower()
    require(not any(pattern in run_log for pattern in ("traceback", "context overflow", "error evaluating user")), "runtime error marker found")
    print(json.dumps({
        "variant": "memory_only",
        "test_rankings": len(test),
        "warmup_rows": len(warmup),
        "credit_updates_during_test": metrics["credit_updates_during_test"],
        "json_fallback_count": run_log.count("error parsing json"),
        "ndcg_at_10": metrics.get("NDCG@10"),
        "wall_time_seconds": metrics.get("wall_time_seconds"),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ACCEPTANCE FAILED: {error}", file=__import__("sys").stderr)
        raise
