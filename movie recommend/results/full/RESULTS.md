# Movie Recommendation Experiment Results

All numbers below are generated automatically from the CSVs in this directory; nothing is filled in or edited by hand.

This is a formal experiment that passed the completeness checks.

- Random seeds: [11, 22, 33, 44, 55].
- Each of the five algorithms is executed independently 5 times; the genre-off branch is trained independently another 5 times, for a total of 30 actual fits.
- Each neural model trains up to 12 epochs, with early stopping on the validation set and restoration of the best weights.
- Uses the full MovieLens 1M data (no subsampling); split roughly 80/10/10 in global chronological order.
- Train/val/test rating rows: 800,164 / 100,024 / 100,021.
- Top-K eligible users: 992; training movie catalog: 3662 movies.
- Hyperparameters are fixed in advance; no tuning. The test set plays no role in tuning.
- Total runtime: 6.3 minutes (CPU; varies by machine).

## Comparison of the five algorithms

| Model | RMSE ↓ | MAE ↓ | Precision@10 ↑ | Recall@10 ↑ | NDCG@10 ↑ |
|---|---|---|---|---|---|
| UserCF | 1.0073 ± 0.0000 | 0.7813 ± 0.0000 | 0.0137 ± 0.0000 | 0.0036 ± 0.0000 | 0.0139 ± 0.0000 |
| ItemCF | 0.9651 ± 0.0000 | 0.7500 ± 0.0000 | 0.0230 ± 0.0000 | 0.0056 ± 0.0000 | 0.0216 ± 0.0000 |
| MF | 0.9142 ± 0.0005 | 0.7177 ± 0.0004 | 0.1585 ± 0.0047 | 0.0510 ± 0.0019 | 0.1779 ± 0.0048 |
| NeuralCF | 0.9248 ± 0.0030 | 0.7243 ± 0.0020 | 0.1027 ± 0.0042 | 0.0356 ± 0.0011 | 0.1131 ± 0.0054 |
| ContentAware | 0.9220 ± 0.0038 | 0.7218 ± 0.0025 | 0.1072 ± 0.0057 | 0.0367 ± 0.0024 | 0.1172 ± 0.0090 |

## Effect of genre features

With genre features added, RMSE changes by -0.0028 (negative = improvement), and NDCG@10 changes by +0.0041 (positive = improvement).

Across the 5 seeds, the genre-aware model beats the baseline on RMSE 3 times and on NDCG@10 3 times. The lowest mean RMSE among the five algorithms is achieved by MF, and the highest mean NDCG@10 by MF.

These conclusions hold only for this data split, hyperparameters, and training budget. Means and standard deviations are not significance tests, and other settings may give different results.

Genre off uses the same random seed and batch order, is independently trained from a freshly initialised model, and does not load baseline weights. ablation_checks.csv records its differences in predictions, parameters, training loss, validation performance, and metrics against NeuralCF; ablation_loss_curves.csv stores the independent training curves, and execution_log.csv records every real fit.

## Sparse-history analysis

Groups use the lowest 25% of training interaction counts: user threshold 41, movie threshold 26.

| Group | NeuralCF RMSE | Genre-aware RMSE | Test rating rows |
|---|---|---|---|
| Sparse users | 0.9527 ± 0.0076 | 0.9566 ± 0.0112 | 12499 |
| Sparse movies | 1.0306 ± 0.0050 | 1.0234 ± 0.0064 | 2412 |

In the sparse-user group, adding genres changes RMSE by +0.0039, i.e. increases (no mean improvement).

In the sparse-movie group, adding genres changes RMSE by -0.0071, i.e. decreases (mean improvement).

The original sparse-user group contains 9 cold-movie rows; the sparse-movie group contains 335 cold-user rows, which use the unified fallback.

To isolate the learnable model, we also report sparse subsets where both sides appear in training:

| Group (both sides warm) | NeuralCF RMSE | Genre-aware RMSE | Test rating rows |
|---|---|---|---|
| Sparse users | 0.9525 ± 0.0076 | 0.9564 ± 0.0112 | 12490 |
| Sparse movies | 1.0211 ± 0.0058 | 1.0127 ± 0.0075 | 2077 |

The two groups may overlap; this is relative sparse-history analysis, not a zero-interaction cold-start experiment.

## Evaluation scope and limitations

- RMSE/MAE cover all test ratings; cold users or movies use the unified training-mean fallback. warm_rmse/warm_mae are also reported in the CSVs for both-sides-warm inspection.
- The global chronological split introduces cold users, so overall rating error can be strongly affected by the unified fallback. See data_summary.json for counts; not all error changes can be attributed to genre features.
- Top-K is evaluated only for users seen in training, over the training catalog, with a positive threshold of 4; full-catalog ranking with no negative sampling. Movies already seen in training and validation are excluded; users without any eligible test positive are not included in the macro average.
- Neighbourhood CF is also refit and evaluated per seed. It has no stochastic training, so its metrics are identical across real runs and its standard deviation is 0.
- All neural models share the same embedding dimension, learning rate, batch size, and early-stopping rule. The genre-aware model has extra genre parameters, so this ablation cannot fully separate the effect of content features from added capacity.

Figures are in figures/; see EXPERIMENT_PROTOCOL.md in the project root for the full reproduction rules.
