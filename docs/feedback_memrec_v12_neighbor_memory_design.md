# FeedbackMemRec V1.2: Stage-R Dynamic Neighbor Memory

## Scope

V1.2 closes one specific loop: a later Stage-R episode may consume long-term
neighbor text committed by an earlier Stage-W episode. It does not change the
memory schema, user-item graph, evidence-credit update, leave-one-facet-out
attribution, read-credit multiplier, write gates, candidates, model, facet
count, or the 1,800-token Stage-R budget.

The long-term state remains:

- `MemoryStorage.user_profiles` and `MemoryStorage.item_descriptions`;
- `EvidenceCreditStore`;
- the existing event logs.

Dynamic neighbor text is copied into an episode-local, read-only snapshot. The
packer cannot mutate `MemoryStorage`, and a Stage-W write can only affect a
later episode.

## Read path

```text
UserItemGraph
  -> existing pruner (k=16, including read-credit adjustment)
  -> existing V1.1 packer path freezes packed IDs/order/static snippets
  -> snapshot memories only for pruner-selected neighbors
  -> top <=4 user and <=4 item memories by final selection score
  -> deterministic Qwen-tokenizer head-tail truncation
  -> append memory only to an already-packed neighbor
  -> Stage-R
```

Missing memories are skipped. V1.2 never traverses additional graph neighbors
to fill the quota. Candidate dynamic memories that are not in the frozen
baseline packed set are logged and dropped.

## Configuration

```yaml
neighbor_memory_read:
  enabled: false
  max_user_neighbors: 4
  max_item_neighbors: 4
  per_memory_tokens: 64
  total_memory_tokens: 512
  preserve_static_neighbors: true
  truncation: head_tail
  head_tokens: 40
  tail_tokens: 24
  persist_events: true
  persist_packed_context: true
```

When `enabled=false`, the packer returns the original V1.1 result without
loading a tokenizer or reading neighbor memories. It may write a baseline
packed-context audit record, but creates no dynamic-memory snapshot, event,
counter, or warmup memory snapshot.

## Frozen baseline and budgets

The existing V1.1 pack is executed first. These values are frozen before any
dynamic text is considered:

- packed neighbor IDs and order;
- static snippets;
- target-user memory and candidate sections;
- baseline context text and exact Qwen token count.

The available dynamic budget is:

```text
remaining_tokens = max(1800 - exact_baseline_context_tokens, 0)
dynamic_budget = min(512, remaining_tokens)
```

Each memory is capped at 64 Qwen tokens. A longer memory retains its head and
tail with ` … ` as an explicit omission marker; marker tokens count toward the
cap. Tail tokens are reduced first if the rendered text exceeds the limit.
Global shortage can only shorten or drop added dynamic memory. Runtime
assertions enforce:

```text
single included memory <= 64 tokens
episode dynamic memory <= 512 tokens
complete packed context <= 1800 tokens
```

The original snippet is retained exactly and the optional block is appended as:

```text
<existing V1.1 user/item neighbor snippet>
Persistent memory:
<token-aware truncated memory>
```

## Audit artifacts

With V1.2 enabled, the run writes:

- `feedback_memrec_logs/neighbor_memory_read_events.jsonl`: one candidate
  memory per row, including full/included hashes, token counts and drop reason;
- `feedback_memrec_logs/packed_context_events.jsonl`: one Stage-R episode per
  row, including pruned/packed/dynamic IDs, frozen snippets, token counts,
  context hashes, configuration and optionally the full context text;
- `feedback_memrec_logs/read_after_write_events.jsonl`: source write and later
  consumer read hashes;
- `memory_after_warmup.jsonl`: emitted after warmup and before test;
- `per_user_metrics.jsonl`: includes `phase=test` and user-level ranks/metrics.

Only a trace whose written and consumed full-memory hashes match is counted as
a successful read-after-write. These artifacts are observational and are never
written back into long-term memory.

## Files changed for V1.2

- `src/memory/packer.py`
- `src/models/memrec_agent.py`
- `src/train/trainer_memrec.py`
- `src/memory/feedback_controller.py`
- this design document

## Explicit non-goals

V1.2 does not add request caching, cross-run LLM-output alignment, memory
versioning, rollback, LLM compression, a global memory pool, a vector database,
training, or fine-tuning. No 20-, 100-, or 300-user experiment is part of this
change.
