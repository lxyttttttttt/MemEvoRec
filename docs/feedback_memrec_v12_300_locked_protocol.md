# FeedbackMemRec V1.2: Locked 300-user Dual-GPU Protocol

Locked before launch on 2026-08-26 UTC. No result-dependent user selection,
threshold change, retry, or protocol change is permitted after queue start.

## Repository and V1.2 state

- Project: `/root/autodl-tmp/MemRec`
- Git HEAD: `58d9031ed91c623b8034d2bf04f39aa937424c33`
- V1.1/V1.2 are present as an uncommitted dirty worktree; `git diff --check`
  passed before lock.
- The only post-smoke code addition is the observational
  `persist_memory_after_warmup` switch required to save the pre-test storage
  snapshot for all three formal variants. It does not enter prompts, candidate
  generation, credit, attribution, gating, or writes.
- Locked algorithm/config/source hashes are stored in
  `docs/feedback_memrec_v12_300_sha256.txt` and are checked before the queue and
  before every individual run.

## Locked users and targets

- User list: `data/eval_user_samples/strict_books_v12_300_seed45.json`
- File SHA-256:
  `41debfbf74d3cf1ccbd0a2eeaa3bf7c1932bde43734967eb7b852094355c41f8`
- Sampling: `random.Random(45).sample(sorted(eligible), 300)`
- Valid Books test users: 7,377
- Excluded union: 320 users
- Eligible after exclusion: 7,057 users
- Selected: 300 unique users, all in the valid test set
- Ordered `(user_id,target_item)` canonical JSON SHA-256:
  `4e021400e09d3e0d696b338f74d410ccafa4b64604e2a4301e2393228c3ff0af`

Intersection proof:

| Excluded set | Size | Intersection with V1.2 300 |
|---|---:|---:|
| Development 5-user | 5 | 0 |
| Development 20-user | 20 | 0 |
| V1.1 Block 1 | 100 | 0 |
| V1.1 Block 2 | 100 | 0 |
| Deferred V1.1 Block 3 | 100 | 0 |

The 5-user set is contained in the 20-user set, so the excluded union is 320,
not 325. All variants load the same locked file and preserve its order. Targets
are the last timestamp-sorted interaction used by the unchanged leave-one-out
dataset split.

## Locked variants

| Variant | Endpoint | Neighbor memory | Credit | Read credit | Write gate | Attribution | Config SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| Corrected | `127.0.0.1:8000` | off | off | off | off | off | `db94c14ac80e667e5f89fd5cc053e72874195322a94fb73047342346506b4469` |
| Closed-loop Read | `127.0.0.1:8001` | on | on | on | off | on | `1b54eb812aa15a982459e9dff9958ad4d04bb897fb97cf2ea5034a851a5d0f5c` |
| Closed-loop Full V1.2 | `127.0.0.1:8000` | on | on | on | immediate + historical | on | `fd7337043319aa564df861fa8224b6c13472c3c697d8adbce81dcc0e91975635` |

Shared frozen settings:

```text
seed=45
n_eval_users=300
n_eval_candidates=10
warmup.enabled=true
warmup.rounds=1
k=16
n_facets=7
temperature=0
tau_tokens=1800
per_memory_tokens=64
total_memory_tokens=512
feedback_credit.freeze_credit=true
serial execution only
```

Each process constructs a new `MemRecAgent`, `MemoryStorage`, and empty
`EvidenceCreditStore`. No load path or writable state is shared.

## GPU and service mapping

```text
Physical GPU 0 / port 8000: Corrected -> Full V1.2
Physical GPU 1 / port 8001: Closed-loop Read
```

Both vLLM services use the same local model directory and these settings:

```text
dtype=half
max-model-len=8192
gpu-memory-utilization=0.88
seed=42
enforce-eager
served-model-name=Qwen2.5-7B-Instruct
```

Client configurations fix temperature to zero. Services do not use tensor
parallelism. Local traffic bypasses all configured HTTP proxies.

## Actual run commands

GPU 0 Corrected:

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/memrec/bin/python -u scripts/run_train.py \
  --model memrec_agent --dataset instructrec-books \
  --config configs/feedback_memrec_books_v12_300_corrected.yaml \
  --device cuda:0 \
  --output_dir outputs/feedback_memrec_v12_continuous_300_corrected
```

GPU 1 Closed-loop Read:

```bash
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/memrec/bin/python -u scripts/run_train.py \
  --model memrec_agent --dataset instructrec-books \
  --config configs/feedback_memrec_books_v12_300_read.yaml \
  --device cuda:0 \
  --output_dir outputs/feedback_memrec_v12_continuous_300_read
```

GPU 0 Full, only after Corrected exits zero and passes acceptance:

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/memrec/bin/python -u scripts/run_train.py \
  --model memrec_agent --dataset instructrec-books \
  --config configs/feedback_memrec_books_v12_300_full.yaml \
  --device cuda:0 \
  --output_dir outputs/feedback_memrec_v12_continuous_300_full
```

The persistent controller is
`scripts/run_feedback_memrec_v12_300_dual_gpu.sh`. It checks locked hashes,
service health, output nonexistence, reports every 50 warmup/test users, records
both GPUs every 30 seconds, never retries, and stops only the failed GPU's
subsequent queue. The independent queue on the other GPU is allowed to finish.

## Fail-closed acceptance

Each run must pass `scripts/validate_feedback_memrec_v12_300_run.py`. After all
three runs, candidate manifests must be byte-identical. Any violation of the
300/300 ranking count, frozen test credit, direct-write preservation, token
caps, static-snippet preservation, hash matching, locked hashes, or serial
protocol invalidates the affected formal comparison and prevents silent final
aggregation.

At the first-hour audit, before any run completed, the validator's log-text
rule was corrected to match the requested frozen failure handling: model JSON
fallbacks are counted and reported, while completed rankings remain eligible.
Structural warmup/test failures, traceback, context overflow and incomplete
rankings remain fail-closed. No user list, experiment config, algorithm source,
running process or output was changed.
