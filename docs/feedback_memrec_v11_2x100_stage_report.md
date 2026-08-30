# FeedbackMemRec V1.1 2×100 Stage Report

Block 1 was preserved and Block 2 was newly run. Block 3 is **deferred/not run**.

## Block results

| Block | Seed | Corrected NDCG@10 | Read NDCG@10 | Full NDCG@10 | Full−Read | Full−Corrected |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 0.689771 | 0.696837 | 0.709473 | +0.012636 | +0.019702 |
| 2 | 43 | 0.700578 | 0.705433 | 0.705433 | +0.000000 | +0.004856 |

## Direction and paired inference

Full exceeds Read in 1/2 blocks and Corrected in 2/2 blocks. This direction count is not a significance test.

- full_minus_read: Block 2 n=100, mean NDCG@10 delta `+0.000000`, bootstrap 95% CI `[+0.000000, +0.000000]`, two-sided sign-flip p=`1.000000`. Common-success sensitivity n=100, mean `+0.000000`, CI `[+0.000000, +0.000000]`, p=`1.000000`.
- full_minus_corrected: Block 2 n=100, mean NDCG@10 delta `+0.004856`, bootstrap 95% CI `[-0.000289, +0.013691]`, two-sided sign-flip p=`0.497250`. Common-success sensitivity n=100, mean `+0.004856`, CI `[-0.000289, +0.014618]`, p=`0.505249`.

## Integrity

- Candidate manifests byte-identical: Block 1 `True`, Block 2 `True`.
- `credit_updates_during_test=0` for all six retained runs: `True`.
- Block 2 handled JSON fallback events: `6` (reranker/test/user 3832: 3; stage_w/warmup/user 3347: 3); fatal transport errors: `0`. These fallbacks did not abort ranking or memory-update control flow.
- Block 1 Full: rejects 45/448 (10.04%); direct writes preserved 348/348 (100.00%); invalid relations 87/2064 (4.22%).
- Block 2 Full: rejects 47/438 (10.73%); direct writes preserved 350/350 (100.00%); invalid relations 149/2305 (6.46%).

## Evidence limitations

- Block 1 lacks recoverable per-user predicted rankings, so paired bootstrap/sign-flip inference uses only Block 2's 100 paired users.
- Two blocks show direction but cannot establish broad cross-block stability.
- Block 3 was intentionally deferred; no missing result was reconstructed, estimated, or imputed.

Complete Hit metrics, runtime/token fields, hashes, gate statistics, and test outputs are in `outputs/feedback_memrec_v11_2x100_summary.json` and `.csv`.
