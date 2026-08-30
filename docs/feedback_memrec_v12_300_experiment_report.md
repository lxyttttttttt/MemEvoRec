# FeedbackMemRec V1.2 300-user Experiment Report

## Bottom line

The four locked runs completed 300 warmup and 300 test users. Each run had
4-5 isolated incomplete LLM rankings, so strict 300/300 acceptance failed.
The prespecified all-300 analysis counts those failures as zero; common-success
analyses exclude only the failed pair members. No primary NDCG comparison is
statistically distinguishable from zero at the 95% level.

## All-300 metrics (ranking failures count as zero)

| Variant | Success | Hit@1 | Hit@3 | Hit@5 | Hit@10 | NDCG@10 | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| corrected | 296/300 | 0.4167 | 0.6500 | 0.7600 | 0.9867 | 0.669025 | 3.16 h |
| memory_only | 295/300 | 0.4300 | 0.6567 | 0.7533 | 0.9833 | 0.671417 | 3.03 h |
| read | 295/300 | 0.4300 | 0.6600 | 0.7667 | 0.9833 | 0.674103 | 7.33 h |
| full | 295/300 | 0.4233 | 0.6433 | 0.7567 | 0.9833 | 0.668487 | 7.37 h |

## Primary incremental comparisons

| Comparison | all-300 NDCG delta | bootstrap 95% CI | sign-flip p | common-success n | common-success delta | common 95% CI | common p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Memory Only - Corrected | +0.002392 | [-0.020216, +0.025441] | 0.8354 | 294 | +0.004720 | [-0.017127, +0.026777] | 0.6746 |
| Read - Memory Only | +0.002686 | [-0.014121, +0.019601] | 0.7617 | 293 | +0.002799 | [-0.011715, +0.017021] | 0.6961 |
| Full - Read | -0.005616 | [-0.029963, +0.018461] | 0.6580 | 292 | -0.004751 | [-0.024595, +0.015713] | 0.6393 |

## Common success across all four variants

All four variants succeeded for 291/300 users.

| Variant | Hit@1 | Hit@3 | Hit@5 | Hit@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| corrected | 0.4227 | 0.6632 | 0.7732 | 1.0000 | 0.679396 |
| memory_only | 0.4364 | 0.6667 | 0.7663 | 1.0000 | 0.682058 |
| read | 0.4364 | 0.6701 | 0.7801 | 1.0000 | 0.684876 |
| full | 0.4296 | 0.6564 | 0.7732 | 1.0000 | 0.680109 |

## Mechanism and fairness audit

- Candidate manifests are byte-identical: `62fb797bc16ccf701a385b456bd657efac7cb592e12a852ef55765c33b7fddc3`.
- Test credit updates are zero in all four variants.
- All locked source/config hashes still pass after execution.
- Read, Full, and Memory Only all obey the 64-token single-memory, 512-token
  dynamic-memory, and 1800-token Stage-R caps; static snippets are preserved.
- Read-after-write mismatches: corrected 0, memory_only 0, read 0, full 0.
- Full gate: 145/1341 collaborative writes rejected (10.81%); item neighbors 102, user neighbors 43.
- All 1052 Full direct-write events were preserved; all immediate rejects had raw_episode_delta < 0.
- Full dynamic memories: 3099 packed, 0 dropped; context tokens min/mean/max 778/1302.8/1586.

## Interpretation

- Dynamic neighbor memory alone is directionally above Corrected, but the CI
  crosses zero; this run does not establish a reliable gain.
- Adding credit and attribution on top of Memory Only is also directionally
  positive, but statistically inconclusive.
- The write gate is directionally negative versus Read in this seed. Its 145
  immediate rejects changed long-term state and context, but did not improve
  aggregate ranking quality in this run.
- These are one dataset, one locked 300-user sequence, and one model seed. The
  three 100-user segments are temporal diagnostics, not independent seeds.

## Historical V1.1 context (separate protocol)

V1.1 Block 1 NDCG@10 was 0.689771/0.696837/0.709473 for
Corrected/Read/Full; Block 2 was 0.700578/0.705433/0.705433.
Thus V1.1 showed positive direction versus Corrected in both blocks, while
the Full-Read gain appeared only in Block 1. These results are not pooled with
V1.2 because V1.2 changes the Stage-R input by reading dynamic neighbor memory.

## Historical and development run inventory

Development rows below are implementation checks, not formal effect
estimates. V1.1 Blocks 1 and 2 are independent 100-user formal blocks.

| Version | Scope | Variant | Users | Hit@1 | Hit@3 | Hit@5 | Hit@10 | NDCG@10 | Wall time |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| V1.0 | development | corrected | 20 | 0.6000 | 0.7000 | 0.7500 | 1.0000 | 0.760049 | 0.22 h |
| V1.0 | development | read | 20 | 0.6000 | 0.7000 | 0.8000 | 1.0000 | 0.763020 | 0.52 h |
| V1.0 | development | full | 20 | 0.6000 | 0.7000 | 0.8000 | 1.0000 | 0.763020 | 0.52 h |
| V1.1 | development | full | 20 | 0.6000 | 0.7000 | 0.8000 | 1.0000 | 0.762126 | 0.52 h |
| V1.1 | formal block 1 | corrected | 100 | 0.4500 | 0.6900 | 0.7200 | 1.0000 | 0.689771 | 1.08 h |
| V1.1 | formal block 1 | read | 100 | 0.4700 | 0.6800 | 0.7400 | 1.0000 | 0.696837 | 2.54 h |
| V1.1 | formal block 1 | full | 100 | 0.4900 | 0.7000 | 0.7400 | 1.0000 | 0.709473 | 2.54 h |
| V1.1 | formal block 2 | corrected | 100 | 0.4700 | 0.6600 | 0.7600 | 1.0000 | 0.700578 | 1.11 h |
| V1.1 | formal block 2 | read | 100 | 0.4800 | 0.6600 | 0.7600 | 1.0000 | 0.705433 | 2.62 h |
| V1.1 | formal block 2 | full | 100 | 0.4800 | 0.6600 | 0.7600 | 1.0000 | 0.705433 | 2.63 h |

## Artifacts

- Summary: `outputs/feedback_memrec_v12_300_summary.json`
- Per-user table: `outputs/feedback_memrec_v12_300_per_user.csv`
- Paired statistics: `outputs/feedback_memrec_v12_300_paired_statistics.json`
