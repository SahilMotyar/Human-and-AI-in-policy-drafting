# Sarvam-only Report

Data source: result/master_dataframe_with_stage5_scores.csv

## Overview
- sarvam_total_rows: 81
- sarvam_valid_rows_excl_refusal_under3kb: 68
- sarvam_refusal_detected: 9
- sarvam_under_3KB: 13
- sarvam_other_issue_nonempty: 68
- sarvam_missing_files: 0
- sarvam_short_lt_500: 13

## Counts by domain and condition
| domain            | condition     |   count |
|:------------------|:--------------|--------:|
| Air Pollution     | Innovation    |       8 |
| Air Pollution     | Status Quo    |       9 |
| Air Pollution     | Unconstrained |       9 |
| Data Protection   | Innovation    |       8 |
| Data Protection   | Status Quo    |      10 |
| Data Protection   | Unconstrained |       8 |
| National Security | Innovation    |      10 |
| National Security | Status Quo    |       9 |
| National Security | Unconstrained |      10 |

## Issues value counts (Sarvam)
| issues                                       |   count |
|:---------------------------------------------|--------:|
| none                                         |      68 |
| under_3KB|refusal_detected                   |       5 |
| under_3KB|refusal_detected|very_short_output |       4 |
| under_3KB|very_short_output                  |       2 |
| under_3KB                                    |       2 |

## National Security short/refusal rates (<500 words)
| condition     |   total |   short_lt_500 |   complete |   refusal_rate_pct |
|:--------------|--------:|---------------:|-----------:|-------------------:|
| Innovation    |      10 |              4 |          6 |               40   |
| Status Quo    |       9 |              1 |          8 |               11.1 |
| Unconstrained |      10 |              7 |          3 |               70   |

## Short outputs (<500 words)
| domain            | condition     | file                                                                                         |   word_count | issues                                       |
|:------------------|:--------------|:---------------------------------------------------------------------------------------------|-------------:|:---------------------------------------------|
| Air Pollution     | Status Quo    | E:\Research paper\output\sarvam_policy_drafts\Air status quo iteration 9 sarvam.txt          |           88 | under_3KB|refusal_detected|very_short_output |
| National Security | Innovation    | E:\Research paper\output\sarvam_policy_drafts\National innovation iteration 1 sarvam.txt     |           98 | under_3KB|refusal_detected|very_short_output |
| National Security | Innovation    | E:\Research paper\output\sarvam_policy_drafts\National innovation iteration 7 sarvam.txt     |          113 | under_3KB|refusal_detected                   |
| National Security | Innovation    | E:\Research paper\output\sarvam_policy_drafts\National innovation iteration 6 sarvam.txt     |          118 | under_3KB|refusal_detected                   |
| National Security | Innovation    | E:\Research paper\output\sarvam_policy_drafts\National innovation iteration 2 sarvam.txt     |          137 | under_3KB|refusal_detected                   |
| National Security | Status Quo    | E:\Research paper\output\sarvam_policy_drafts\National status quo iteration 10 sarvam.txt    |          100 | under_3KB|refusal_detected                   |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 7 sarvam.txt  |           91 | under_3KB|refusal_detected|very_short_output |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 1 sarvam.txt  |           93 | under_3KB|refusal_detected|very_short_output |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 4 sarvam.txt  |           95 | under_3KB|very_short_output                  |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 6 sarvam.txt  |           97 | under_3KB|very_short_output                  |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 10 sarvam.txt |          102 | under_3KB|refusal_detected                   |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 9 sarvam.txt  |          126 | under_3KB                                    |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 5 sarvam.txt  |          127 | under_3KB                                    |

## Refusal-detected outputs
| domain            | condition     | file                                                                                         |   word_count | issues                                       |
|:------------------|:--------------|:---------------------------------------------------------------------------------------------|-------------:|:---------------------------------------------|
| Air Pollution     | Status Quo    | E:\Research paper\output\sarvam_policy_drafts\Air status quo iteration 9 sarvam.txt          |           88 | under_3KB|refusal_detected|very_short_output |
| National Security | Innovation    | E:\Research paper\output\sarvam_policy_drafts\National innovation iteration 1 sarvam.txt     |           98 | under_3KB|refusal_detected|very_short_output |
| National Security | Innovation    | E:\Research paper\output\sarvam_policy_drafts\National innovation iteration 2 sarvam.txt     |          137 | under_3KB|refusal_detected                   |
| National Security | Innovation    | E:\Research paper\output\sarvam_policy_drafts\National innovation iteration 6 sarvam.txt     |          118 | under_3KB|refusal_detected                   |
| National Security | Innovation    | E:\Research paper\output\sarvam_policy_drafts\National innovation iteration 7 sarvam.txt     |          113 | under_3KB|refusal_detected                   |
| National Security | Status Quo    | E:\Research paper\output\sarvam_policy_drafts\National status quo iteration 10 sarvam.txt    |          100 | under_3KB|refusal_detected                   |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 1 sarvam.txt  |           93 | under_3KB|refusal_detected|very_short_output |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 10 sarvam.txt |          102 | under_3KB|refusal_detected                   |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 7 sarvam.txt  |           91 | under_3KB|refusal_detected|very_short_output |

## Under_3KB outputs
| domain            | condition     | file                                                                                         |   word_count | issues                                       |
|:------------------|:--------------|:---------------------------------------------------------------------------------------------|-------------:|:---------------------------------------------|
| Air Pollution     | Status Quo    | E:\Research paper\output\sarvam_policy_drafts\Air status quo iteration 9 sarvam.txt          |           88 | under_3KB|refusal_detected|very_short_output |
| National Security | Innovation    | E:\Research paper\output\sarvam_policy_drafts\National innovation iteration 1 sarvam.txt     |           98 | under_3KB|refusal_detected|very_short_output |
| National Security | Innovation    | E:\Research paper\output\sarvam_policy_drafts\National innovation iteration 2 sarvam.txt     |          137 | under_3KB|refusal_detected                   |
| National Security | Innovation    | E:\Research paper\output\sarvam_policy_drafts\National innovation iteration 6 sarvam.txt     |          118 | under_3KB|refusal_detected                   |
| National Security | Innovation    | E:\Research paper\output\sarvam_policy_drafts\National innovation iteration 7 sarvam.txt     |          113 | under_3KB|refusal_detected                   |
| National Security | Status Quo    | E:\Research paper\output\sarvam_policy_drafts\National status quo iteration 10 sarvam.txt    |          100 | under_3KB|refusal_detected                   |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 1 sarvam.txt  |           93 | under_3KB|refusal_detected|very_short_output |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 10 sarvam.txt |          102 | under_3KB|refusal_detected                   |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 4 sarvam.txt  |           95 | under_3KB|very_short_output                  |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 5 sarvam.txt  |          127 | under_3KB                                    |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 6 sarvam.txt  |           97 | under_3KB|very_short_output                  |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 7 sarvam.txt  |           91 | under_3KB|refusal_detected|very_short_output |
| National Security | Unconstrained | E:\Research paper\output\sarvam_policy_drafts\National unconstrained iteration 9 sarvam.txt  |          126 | under_3KB                                    |

## File coverage check
- Draft files on disk: 81
- Draft files referenced in master df: 81
- In output dir but not in master df: 0
- In master df but missing on disk: 0

## Charts
- sarvam_counts_by_domain_condition.png
- sarvam_ns_short_rate.png
- sarvam_wordcount_hist.png
