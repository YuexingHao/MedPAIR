# Concordance Summary Tables

## Table 1. Model Ranking
| Rank | Model | Avg Jaccard vs Others (%) | Jaccard vs Human (%) |
| --- | --- | --- | --- |
| 1 | Qwen-72B (SR) | 57.7 | 63.48 |
| 2 | MedGemma-27B (SR) | 56.81 | 61.01 |
| 3 | Human | 54.89 | 100 |
| 4 | GPT-5 (SR) | 53.99 | 61.15 |
| 5 | Llama-70B (SR) | 53.89 | 54.6 |
| 6 | Qwen-14B (CC) | 47.18 | 57.84 |
| 7 | Llama-70B (CC) | 46.27 | 55.73 |
| 8 | Qwen-72B (CC) | 46.23 | 55.8 |
| 9 | Qwen-14B (SR) | 45.59 | 43.99 |
| 10 | GPT-4o (SR) | 40.29 | 40.45 |

## Table 2. Top/Bottom Pairwise Concordance
| Section | Model A | Model B | N Origins | Mean Jaccard (%) | SD (%) | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| Top 10 | MedGemma-27B (SR) | Qwen-72B (SR) | 623 | 72.57 | 23.42 | [70.67, 74.36] |
| Top 10 | MedGemma-27B (SR) | Llama-70B (SR) | 621 | 70.03 | 23.88 | [68.18, 71.88] |
| Top 10 | Qwen-72B (SR) | Llama-70B (SR) | 622 | 69 | 23.87 | [67.11, 70.91] |
| Top 10 | GPT-5 (SR) | MedGemma-27B (SR) | 623 | 65.53 | 25.08 | [63.56, 67.53] |
| Top 10 | GPT-5 (SR) | Qwen-72B (SR) | 623 | 64 | 24.84 | [62.09, 65.94] |
| Top 10 | Human | Qwen-72B (SR) | 623 | 63.48 | 22.59 | [61.66, 65.17] |
| Top 10 | Qwen-14B (SR) | Llama-70B (SR) | 622 | 61.75 | 30.1 | [59.32, 64.07] |
| Top 10 | Human | GPT-5 (SR) | 623 | 61.15 | 23.9 | [59.35, 63.03] |
| Top 10 | Human | MedGemma-27B (SR) | 623 | 61.01 | 23.15 | [59.18, 62.83] |
| Top 10 | GPT-5 (SR) | Llama-70B (SR) | 623 | 61.01 | 25.69 | [59.04, 63.01] |
| Bottom 10 | Llama-70B (SR) | Qwen-72B (CC) | 623 | 41.52 | 24.16 | [39.61, 43.45] |
| Bottom 10 | Llama-70B (SR) | Llama-70B (CC) | 623 | 41.33 | 22.85 | [39.49, 43.12] |
| Bottom 10 | Human | GPT-4o (SR) | 623 | 40.45 | 26.32 | [38.32, 42.43] |
| Bottom 10 | GPT-4o (SR) | Qwen-14B (SR) | 622 | 37.33 | 32 | [34.83, 39.78] |
| Bottom 10 | GPT-4o (SR) | Qwen-14B (CC) | 623 | 34.86 | 23.93 | [32.94, 36.76] |
| Bottom 10 | GPT-4o (SR) | Qwen-72B (CC) | 623 | 34.5 | 24.36 | [32.62, 36.44] |
| Bottom 10 | Qwen-14B (SR) | Qwen-14B (CC) | 623 | 34.44 | 22.6 | [32.70, 36.19] |
| Bottom 10 | GPT-4o (SR) | Llama-70B (CC) | 623 | 34.14 | 23.66 | [32.28, 35.92] |
| Bottom 10 | Qwen-14B (SR) | Qwen-72B (CC) | 623 | 34.07 | 23.73 | [32.24, 36.00] |
| Bottom 10 | Qwen-14B (SR) | Llama-70B (CC) | 623 | 33.96 | 22.86 | [32.13, 35.67] |

## Table 3. Human-Alignment Pairwise Tests (sorted by p-value)
| Better Model (vs Human) | Worse Model (vs Human) | Delta Jaccard vs Human (%) | 95% CI of Delta | N Origins | p-value | Sig | Holm 0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen-72B (SR) | GPT-4o (SR) | 23.03 | [20.63, 25.34] | 623 | 6.06e-64 | *** | Yes |
| Qwen-72B (SR) | Qwen-14B (SR) | 19.49 | [17.43, 21.56] | 623 | 2.46e-60 | *** | Yes |
| MedGemma-27B (SR) | GPT-4o (SR) | 20.56 | [18.24, 22.93] | 623 | 8.88e-56 | *** | Yes |
| MedGemma-27B (SR) | Qwen-14B (SR) | 17.03 | [14.99, 18.99] | 623 | 5.76e-51 | *** | Yes |
| GPT-5 (SR) | GPT-4o (SR) | 20.70 | [18.28, 23.21] | 623 | 1.39e-50 | *** | Yes |
| GPT-5 (SR) | Qwen-14B (SR) | 17.17 | [14.85, 19.47] | 623 | 4.44e-41 | *** | Yes |
| Qwen-14B (CC) | GPT-4o (SR) | 17.39 | [14.65, 20.09] | 623 | 2.45e-31 | *** | Yes |
| Llama-70B (SR) | GPT-4o (SR) | 14.14 | [11.75, 16.56] | 623 | 8.42e-29 | *** | Yes |
| Qwen-72B (CC) | GPT-4o (SR) | 15.35 | [12.63, 18.09] | 623 | 6.60e-26 | *** | Yes |
| Llama-70B (SR) | Qwen-14B (SR) | 10.61 | [8.75, 12.49] | 623 | 3.26e-25 | *** | Yes |
| Llama-70B (CC) | GPT-4o (SR) | 15.28 | [12.58, 18.02] | 623 | 8.30e-25 | *** | Yes |
| Qwen-72B (SR) | Llama-70B (SR) | 8.88 | [7.17, 10.58] | 623 | 1.44e-22 | *** | Yes |
| Qwen-14B (CC) | Qwen-14B (SR) | 13.86 | [11.05, 16.48] | 623 | 9.84e-22 | *** | Yes |
| Llama-70B (CC) | Qwen-14B (SR) | 11.74 | [8.97, 14.50] | 623 | 2.41e-16 | *** | Yes |
| Qwen-72B (CC) | Qwen-14B (SR) | 11.82 | [9.06, 14.52] | 623 | 3.33e-16 | *** | Yes |
| MedGemma-27B (SR) | Llama-70B (SR) | 6.42 | [4.67, 8.09] | 623 | 2.43e-13 | *** | Yes |
| GPT-5 (SR) | Llama-70B (SR) | 6.56 | [4.61, 8.45] | 623 | 7.37e-11 | *** | Yes |
| Qwen-72B (SR) | Llama-70B (CC) | 7.75 | [5.44, 10.10] | 623 | 3.72e-10 | *** | Yes |
| Qwen-72B (SR) | Qwen-72B (CC) | 7.67 | [5.21, 10.10] | 623 | 1.75e-09 | *** | Yes |
| Qwen-72B (SR) | Qwen-14B (CC) | 5.63 | [3.23, 8.06] | 623 | 7.54e-06 | *** | Yes |

Significance stars: `*** p<0.001`, `** p<0.01`, `* p<0.05`, `ns` otherwise.