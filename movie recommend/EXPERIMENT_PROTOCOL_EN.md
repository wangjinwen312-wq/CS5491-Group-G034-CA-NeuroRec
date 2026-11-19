# Experimental Protocol

This document lists the experimental rules adopted uniformly by the group project. All models follow these rules; rules are not changed ad hoc based on test results.

## 1. Research Question

Under the same data and evaluation conditions, can adding movie genre information to NeuralCF reduce rating prediction error and improve Top-K recommendation? Does the effect differ for users and movies with few training interactions?

The main comparison is between NeuralCF and ContentAware. UserCF, ItemCF, and MF provide reference points for traditional and learnable models. Zero improvement or even degradation is an acceptable outcome; conclusions are based on real experiments.

## 2. Data and Splits

- Use the full official MovieLens 1M dataset, retaining valid explicit ratings on the 1–5 scale.
- Apply a fixed split in ascending global timestamp order, targeting 80% training, 10% validation, and 10% test.
- Ratings sharing the same timestamp are not split across partitions, so the actual proportions may deviate slightly from the targets.
- Training timestamps precede validation timestamps, which precede test timestamps. Random seeds do not re-split the data.
- The mapping from raw user/movie IDs to contiguous indices serves identification only; held-out ratings are never used to estimate parameters.
- Movie titles and genres are static content information and may be used to parse the catalog; genre vectors are learned only from training ratings.
- Normalization, average ratings, similarities, interaction counts, neighbors, and model parameters are estimated from the training set only.

Splitting by global timestamp produces new users or movies absent from training. Their ratings are retained for overall rating error, with their counts and the unified fallback strategy recorded. Ranking evaluation is restricted to users and movies seen in training, and filtering and coverage are reported. The two answer different questions; ranking metrics must not be interpreted as performance over all new users or movies.

## 3. Training Task

MF, NeuralCF, and ContentAware all perform explicit rating regression trained with rating error. Ratings of 1–3 in the training records are retained and are not deleted as missing data.

No unobserved "negative samples" are constructed for regression training. At ranking time, items are simply sorted by predicted rating in descending order. Observed low ratings and unobserved preferences are different kinds of information.

Rating predictions are clipped to [1, 5] for all methods. Ranking uses unclipped predicted scores to avoid additional ties caused by clipping. Cold start follows the unified rules below, with averages computed from the training set only:

| User seen in training | Movie seen in training | Fallback prediction |
|---|---|---|
| Yes | No | The user's training average rating |
| No | Yes | The movie's training average rating |
| No | No | The global training average rating |

The validation set is used for early stopping only; the test set does not participate in training, early stopping, or parameter selection. After training, the training and validation sets are not combined for retraining, and all saved results correspond to the same training boundary.

## 4. Rating Prediction Evaluation

Compute over all currently held-out ratings:

\[
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(r_i-\hat r_i)^2},\qquad
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|r_i-\hat r_i|.
\]

Lower RMSE and MAE are better. Records rated ≥4 must not be evaluated alone. Test-set IDs unseen in training must use the pre-fixed fallback; random or untrained embeddings must not be used for prediction.

## 5. Top-K Evaluation

### 5.1 Likes and Candidate Set

- Movies with a true rating ≥4 are defined as relevant items for that user in the current held-out set.
- The candidate catalog consists of movies appearing in the training set.
- At validation time, movies already rated by the user in the training set are excluded.
- At test time, movies already rated by the user in the training and validation sets are excluded.
- Test target movies must remain in the candidate set; all "movies ever seen" in the full dataset must not be excluded wholesale.
- Relevant items in the current held-out set are intersected with the candidate catalog.
- Full-candidate ranking is used; negative items are not sampled for evaluation. Candidate and exclusion rules are identical across all models.
- Ties in predicted scores are broken by a fixed movie-index order to avoid ambiguous tie ordering.

Only users who appear in training and have a non-empty current relevant set enter the Top-K macro average. Users unseen in training and users with no valid relevant movies are counted separately and must not be silently ignored.

### 5.2 Metrics

For user u, let L(u,K) be the top-K of the recommendation list and R(u) the current relevant set:

- Precision@K = |L(u,K) ∩ R(u)| / K.
- Recall@K = |L(u,K) ∩ R(u)| / |R(u)|.
- DCG@K uses binary relevance; a hit at position j contributes 1/log₂(j+1), with j starting at 1.
- IDCG@K is the DCG when the min(K, |R(u)|) relevant movies are placed first.
- NDCG@K = DCG@K / IDCG@K.

Metrics are computed per valid user first, then averaged with equal weight over users. The official report uses K = 5, 10, and 20. Users are not weighted by interaction count, and user-level macro averaging is not mixed with interaction-level micro averaging.

Unobserved movies lack true preference labels; this protocol performs offline retrieval evaluation on held-out positives only. Full-candidate numbers cannot be directly compared with literature results based on "one positive plus several sampled negatives."

## 6. Fixed Parameters and Repeated Experiments

- All models use the same fixed split, candidate set, and evaluation implementation.
- To keep the course project manageable, parameters are fixed in advance; the hyperparameter search budget for every algorithm is 0.
- Neural models share the same 32-dimensional embedding, learning rate 0.003, training batch size 8192, and at most 12 epochs, with the same validation early-stopping rule and patience of 3 epochs.
- Neighbor counts and similarity shrinkage parameters for UserCF and ItemCF are also fixed in advance and stored in the configuration. Algorithms need not share parameter names or training cost.
- Parameters are not repeatedly modified based on test performance. Fixed parameters do not guarantee that every algorithm reaches its optimum; this limitation is noted in the results analysis.
- Five training random seeds are used: 11, 22, 33, 44, 55.
- Each of the five algorithms recreates, fits, and evaluates a model per seed (25 real runs), plus 5 independent GenreOff trainings, for 30 fits in total. Every run is recorded in `execution_log.csv`; a single fit's results are not duplicated as multiple runs.
- For neural models, fixed seeds control initialization and training sample order; the early-stopping rule and chosen epoch are recorded. Debug mode uses one seed and at most two epochs and cannot replace the official five-seed results.
- UserCF and ItemCF also run independently per seed. Because the algorithms are deterministic, identical results under fixed data are expected; a standard deviation of 0 is normal, and data must not be perturbed merely to produce a nonzero standard deviation.
- The final summary reports mean, sample standard deviation (ddof=1), and the number of valid repeats n. A single-seed debug experiment has no estimable sample standard deviation.

## 7. Genre Ablation

Genre On and Genre Off share the same model implementation. Disabling the genre branch restores the NeuralCF architecture, initialization rule, and training settings. GenreOff is initialized from scratch and trained independently for each seed, using the same sample order and early-stopping rule as NeuralCF; it does not load baseline-trained weights.

Per-epoch training records of the independent GenreOff are saved to `ablation_loss_curves.csv`. After training, its parameters, predictions, metrics, and training curves are compared with NeuralCF per seed, and the differences are written to `ablation_checks.csv`. This check verifies whether independent retraining reproduces the baseline; equality after loading the same weights does not substitute for it.

`ablation.csv` keeps all rating, Top-K, and grouped metrics for Genre On/Off; `ablation_summary.csv` summarizes mean, sample standard deviation, and valid counts by On/Off status and metric. Figures focus on RMSE, MAE, and NDCG@10. The maximum prediction difference between the disabled variant and NeuralCF in figures refers to the independently trained comparison. The genre model may have more parameters, so ablation results describe the gain within this implementation and cannot automatically rule out the effect of changed model capacity; a parameter-matched comparison can be added as further research.

## 8. Sparse-History Groups

Count the number of ratings per user and per movie in training separately. Compute the lower 25th percentile over nonzero training interaction counts only; users/movies whose interaction count does not exceed the threshold define the corresponding sparse group.

- Sparse user group: held-out ratings whose user belongs to the sparse user set.
- Sparse movie group: held-out ratings whose movie belongs to the sparse movie set.
- The two groups may overlap; report RMSE, MAE, and sample size separately for each.
- Thresholds and group sets are defined and saved from training data; grouping by full-data counts is not allowed, and held-out sample sizes are counted separately.
- Zero-history cases of new users/movies are recorded separately and are not mixed into the relative-sparse group definition.

Users in the sparse user group have training history, but the corresponding movie may still be unseen in training; likewise, movies in the sparse movie group may correspond to users unseen in training. Therefore, in addition to the original grouped metrics, ratings where the other side is cold-start are recorded, and a warm-only group (both sides seen in training) RMSE, MAE, and sample size are computed separately. The original group describes overall performance; the warm-only results help judge whether the unified average fallback affects conclusions; the two are not mixed.

MovieLens 1M original users have at least 20 ratings, so this grouping reflects relatively sparse training history after splitting and must not be called a strict zero-interaction cold-start experiment.

## 9. Output and Verification

Retain raw metrics for every model and seed, along with data provenance, configuration, split statistics, cold-start coverage, and training logs. All eight figure types are read from CSVs; results are not filled in manually.

Before completion, check that the official configuration has at least 5 distinct seeds, that execution records for all five models and the independent GenreOff are complete, and that all required metrics are finite values; single-seed quick mode is for debugging only. Any failed check raises an error and does not write the `complete` status.

Error bars are cross-seed sample standard deviation reflecting run-to-run randomness; under a fixed split, cross-seed standard deviation does not include data-split uncertainty and is not equivalent to a confidence interval. Training curves aggregate over seeds that still have records at each epoch; early stopping changes sample counts later in the run.

The final discussion addresses whether rating and ranking metrics agree, whether improvements are stable, whether sparse groups benefit, and how candidate scope and cold-start fallback limit the conclusions.

> （注：部分内容由豆包 AI 生成）