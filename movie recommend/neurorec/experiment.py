"""Compare the five algorithms on the same data, save real metrics, and generate assignment figures."""
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
import torch

from .data import load_data, summarize
from .metrics import evaluate_rating, prepare_ranking, ranking_metrics
from .models import UserCF, ItemCF
from .plots import plot_all
from .training import (fit_neural, rating_predict, raw_predict,
                       set_seed, training_means)


# The five main algorithms and the genre-off ablation branch are recorded separately
# to avoid mixing comparisons.
MODELS = ['UserCF', 'ItemCF', 'MF', 'NeuralCF', 'ContentAware']
CHECK_DIFFERENCES = ['max_abs_prediction_difference', 'max_abs_parameter_difference',
                     'max_abs_train_loss_difference', 'max_abs_val_rmse_difference',
                     'max_abs_metric_difference']


def required_metrics(config):
    """List all required metrics, shared by result validation and ablation consistency checks."""
    metrics = ['rmse', 'mae']
    metrics += [f'{metric}@{k}' for k in config['ks'] for metric in ['precision', 'recall', 'ndcg']]
    metrics += [f'{group}_{metric}' for group in
                ['sparse_user', 'sparse_item', 'warm', 'sparse_user_warm', 'sparse_item_warm']
                for metric in ['rmse', 'mae']]
    return metrics


def validate_seeds(config):
    """Reject duplicate seeds or debug runs with fewer than five repetitions."""
    seeds = config['seeds']
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Random seeds must be nonempty and distinct')
    if config.get('mode', 'full') not in {'full', 'quick'}:
        raise ValueError('mode must be full or quick')
    if config.get('mode', 'full') == 'full' and len(seeds) < 5:
        raise ValueError('A formal experiment requires at least five distinct seeds')


def validate_results(runs, ablations, checks, execution, config):
    """Only mark the experiment complete if every record is present and valid."""
    validate_seeds(config)
    seeds = set(config['seeds'])
    metrics = required_metrics(config)

    def check_frame(frame, keys, expected, numeric):
        """Verify columns, record combinations, and numeric values; reject missing/duplicate/invalid rows."""
        required = set(keys) | set(numeric)
        if required - set(frame.columns):
            raise ValueError(f'Missing result columns: {sorted(required - set(frame.columns))}')
        actual = set(frame[keys].itertuples(index=False, name=None))
        if frame.duplicated(keys).any() or actual != expected or len(frame) != len(expected):
            raise ValueError(f'Incomplete or duplicate records for {keys}')
        if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
            raise ValueError(f'Non-finite result values in {numeric}')

    check_frame(runs, ['model', 'seed'], {(m, s) for m in MODELS for s in seeds},
                metrics + ['fit_seconds', 'best_epoch'])
    if (runs[['fit_seconds', 'best_epoch']] < 0).any().any():
        raise ValueError('Negative fit time or epoch')
    check_frame(ablations, ['variant', 'seed'],
                {(v, s) for v in ['genre_on', 'genre_off'] for s in seeds},
                metrics + ['max_abs_difference'])
    # The genre-on variant should match the improved model; the independently trained
    # genre-off variant should reproduce the baseline.
    for variant, model in [('genre_on', 'ContentAware'), ('genre_off', 'NeuralCF')]:
        actual = ablations.loc[ablations.variant == variant].set_index('seed').sort_index()
        baseline = runs.loc[runs.model == model].set_index('seed').sort_index()
        if not np.allclose(actual[metrics], baseline[metrics], rtol=0, atol=1e-10):
            raise ValueError(f'{variant} metrics do not match the corresponding model')
    check_frame(checks, ['seed'], {(s,) for s in seeds},
                CHECK_DIFFERENCES + ['best_epoch', 'baseline_best_epoch'])
    if 'independently_trained' not in checks or not checks.independently_trained.eq(True).all():
        raise ValueError('Every genre-off model must be independently trained')
    differences = checks[CHECK_DIFFERENCES].to_numpy(dtype=float)
    if (differences < 0).any() or (differences > 1e-10).any():
        raise ValueError('Independent genre-off training does not reproduce the baseline')
    if not checks.best_epoch.eq(checks.baseline_best_epoch).all():
        raise ValueError('Genre-off and baseline selected different epochs')
    if not ablations.loc[ablations.variant == 'genre_off', 'max_abs_difference'].between(0, 1e-10).all():
        raise ValueError('Genre-off prediction difference exceeds tolerance')
    # Metrics tables alone do not prove training occurred; cross-check independent
    # fits and epoch counts.
    expected_execution = {(m, s, 'main') for m in MODELS for s in seeds}
    expected_execution |= {('GenreOff', s, 'ablation_off') for s in seeds}
    check_frame(execution, ['model', 'seed', 'role'], expected_execution,
                ['fit_seconds', 'epochs', 'best_epoch'])
    if 'run_id' not in execution or execution.run_id.isna().any() or execution.run_id.duplicated().any():
        raise ValueError('Each independent fit must have a unique run ID')
    if 'independent_fit' not in execution or not execution.independent_fit.eq(True).all():
        raise ValueError('Cached result rows cannot replace independent fits')
    if (execution[['fit_seconds', 'epochs', 'best_epoch']] < 0).any().any():
        raise ValueError('Negative fit time or epoch count')
    if not execution.best_epoch.le(execution.epochs).all():
        raise ValueError('Best epoch exceeds executed epochs')
    if not np.equal(execution[['epochs', 'best_epoch']],
                    np.floor(execution[['epochs', 'best_epoch']])).all().all():
        raise ValueError('Epoch counts must be integers')
    for row in execution.itertuples():
        if row.role == 'main':
            recorded = runs.loc[(runs.model == row.model) & (runs.seed == row.seed)].iloc[0]
            if row.best_epoch != recorded.best_epoch:
                raise ValueError('Execution log and main results disagree on the best epoch')
            if row.model in {'UserCF', 'ItemCF'} and (row.epochs != 0 or row.best_epoch != 0):
                raise ValueError('Neighbourhood CF must not claim neural training epochs')
        else:
            recorded = checks.loc[checks.seed == row.seed].iloc[0]
            if row.best_epoch != recorded.best_epoch:
                raise ValueError('Execution log and ablation check disagree on the best epoch')
        if row.model not in {'UserCF', 'ItemCF'} and (row.epochs < 1 or row.best_epoch < 1):
            raise ValueError('Neural models must have actual training epochs')
    for row in checks.itertuples():
        baseline_epoch = runs.loc[(runs.model == 'NeuralCF') & (runs.seed == row.seed), 'best_epoch'].iloc[0]
        if row.baseline_best_epoch != baseline_epoch:
            raise ValueError('Ablation check does not refer to the actual baseline best epoch')
    for metric in metrics:
        if '@' in metric and not runs[metric].between(0, 1).all():
            raise ValueError(f'{metric} is outside [0, 1]')
        if metric.endswith(('rmse', 'mae')) and (runs[metric] < 0).any():
            raise ValueError(f'{metric} cannot be negative')
    return dict(status='passed', formal=config.get('mode', 'full') == 'full',
                model_seed_rows=len(runs), independent_fits=len(execution),
                ablation_rows=len(ablations))


def write_json(path, value):
    """Save config or status as UTF-8 so non-ASCII fields stay readable."""
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def evaluate_ranking(model, ranking, ks):
    """Rank over the full eligible catalog, then average per-user metrics with equal weights."""
    rows = []
    for user in ranking.users:
        items = ranking.catalog
        scores = raw_predict(model, np.full(len(items), user, dtype=np.int64), items)
        seen = ranking.seen.get(int(user), set())
        # Exclude only history observed at evaluation time; keep target movies from
        # the current held-out split.
        available = np.array([int(item) not in seen for item in items])
        candidates, values = items[available], scores[available]
        # Break score ties by movie ID so every run uses the same rule.
        order = np.lexsort((candidates, -values))[:max(ks)]
        rows.append(ranking_metrics(candidates[order], ranking.relevant[int(user)], ks))
    if not rows:
        raise ValueError('No eligible ranking users; check the data split.')
    # Give each valid user equal weight so heavy raters cannot dominate the final metric.
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def sparse_groups(data):
    """Group by training interaction counts only, taking the lowest 25% of non-zero users/movies."""
    user_limit = float(np.quantile(data.train_user_counts[data.train_user_counts > 0], .25))
    item_limit = float(np.quantile(data.train_item_counts[data.train_item_counts > 0], .25))
    uc = data.train_user_counts[data.test.user.to_numpy()]
    ic = data.train_item_counts[data.test.item.to_numpy()]
    masks = {'sparse_user': (uc > 0) & (uc <= user_limit),
             'sparse_item': (ic > 0) & (ic <= item_limit),
             'warm': (uc > 0) & (ic > 0)}
    # Also inspect subsets where both user and movie have training history,
    # to isolate cold-start fallback effects.
    masks['sparse_user_warm'] = masks['sparse_user'] & masks['warm']
    masks['sparse_item_warm'] = masks['sparse_item'] & masks['warm']
    info = {'definition': 'training count > 0 and <= 25th percentile of nonzero counts',
            'user_count_threshold': user_limit, 'item_count_threshold': item_limit,
            'sparse_user_entities': int(((data.train_user_counts > 0) &
                                         (data.train_user_counts <= user_limit)).sum()),
            'sparse_item_entities': int(((data.train_item_counts > 0) &
                                         (data.train_item_counts <= item_limit)).sum()),
            'test_rating_counts': {name: int(mask.sum()) for name, mask in masks.items()},
            'opposite_side_cold_rows': {
                'sparse_user': int((masks['sparse_user'] & (ic == 0)).sum()),
                'sparse_item': int((masks['sparse_item'] & (uc == 0)).sum())}}
    return masks, info


def evaluate_model(model, data, means, ranking, masks, config):
    """Unified evaluation of rating error, Top-K, and sparse groups; all algorithms use the same rules."""
    prediction = rating_predict(model, data.test.user, data.test.item, data, means)
    truth = data.test.rating.to_numpy()
    result = evaluate_rating(truth, prediction)
    result.update(evaluate_ranking(model, ranking, config['ks']))
    for group, mask in masks.items():
        scores = evaluate_rating(truth[mask], prediction[mask]) if mask.any() else {'rmse': np.nan, 'mae': np.nan}
        result.update({f'{group}_{key}': value for key, value in scores.items()})
    return result, prediction


def save_tables(output, rows, histories, ablations, checks, ablation_histories, execution):
    """Save per-run raw records, then compute per-model cross-seed mean and sample SD."""
    runs = pd.DataFrame(rows)
    runs.to_csv(output / 'runs.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(histories).to_csv(output / 'loss_curves.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(ablations).to_csv(output / 'ablation.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(output / 'ablation_checks.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(ablation_histories).to_csv(output / 'ablation_loss_curves.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(execution).to_csv(output / 'execution_log.csv', index=False, encoding='utf-8-sig')
    metrics = [c for c in runs.columns if c not in {'model', 'seed', 'fit_seconds', 'best_epoch'}]
    # ddof=1 selects the sample standard deviation; a single debug run cannot estimate
    # cross-seed variation, so missing values are kept.
    summary = []
    for name, group in runs.groupby('model', sort=False):
        for metric in metrics:
            values = group[metric].dropna()
            summary.append(dict(model=name, metric=metric, mean=values.mean(),
                                std=values.std(ddof=1) if len(values) > 1 else np.nan, n=len(values)))
    pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False, encoding='utf-8-sig')
    if ablations:
        ablation_summary = []
        for variant, group in pd.DataFrame(ablations).groupby('variant', sort=False):
            for metric in metrics:
                values = group[metric].dropna()
                ablation_summary.append(dict(variant=variant, metric=metric, mean=values.mean(),
                                             std=values.std(ddof=1) if len(values) > 1 else np.nan,
                                             n=len(values)))
        pd.DataFrame(ablation_summary).to_csv(output / 'ablation_summary.csv', index=False, encoding='utf-8-sig')


def write_report(output, config, data_info, sparse_info, elapsed):
    """Generate the results write-up from the actual CSVs; describe improvements or regressions from data, no preset conclusions."""
    summary = pd.read_csv(output / 'summary.csv')
    lookup = {(r.model, r.metric): (r.mean, r.std) for r in summary.itertuples()}
    def value(model, metric):
        mean, std = lookup[(model, metric)]
        return f'{mean:.4f} ± {std:.4f}' if np.isfinite(std) else f'{mean:.4f} (single run)'
    lines = ['# Movie Recommendation Experiment Results', '',
             'All numbers below are generated automatically from the CSVs in this directory; nothing is filled in or edited by hand.', '',
             ('This is a formal experiment that passed the completeness checks.'
              if config.get('mode', 'full') == 'full'
              else 'These are debug results; they cannot replace a formal five-seed experiment.'), '',
             f'- Random seeds: {config["seeds"]}.',
             f'- Each of the five algorithms is executed independently {len(config["seeds"])} times; '
             f'the genre-off branch is trained independently another '
             f'{len(config["seeds"])} times, for a total of {6 * len(config["seeds"])} actual fits.',
             f'- Each neural model trains up to {config["epochs"]} epochs, with early stopping on the validation set and restoration of the best weights.',
             '- Uses the full MovieLens 1M data (no subsampling); split roughly 80/10/10 in global chronological order.',
             f'- Train/val/test rating rows: {data_info["partitions"]["train"]["rows"]:,} / '
             f'{data_info["partitions"]["val"]["rows"]:,} / {data_info["partitions"]["test"]["rows"]:,}.',
             f'- Top-K eligible users: {data_info["ranking_test"]["evaluated_users"]}; '
             f'training movie catalog: {data_info["ranking_test"]["catalog_size"]} movies.',
             '- Hyperparameters are fixed in advance; no tuning. The test set plays no role in tuning.',
             f'- Total runtime: {elapsed / 60:.1f} minutes (CPU; varies by machine).', '',
             '## Comparison of the five algorithms', '',
             '| Model | RMSE ↓ | MAE ↓ | Precision@10 ↑ | Recall@10 ↑ | NDCG@10 ↑ |',
             '|---|---|---|---|---|---|']
    for model in ['UserCF', 'ItemCF', 'MF', 'NeuralCF', 'ContentAware']:
        lines.append('| ' + model + ' | ' + ' | '.join(value(model, metric) for metric in
                     ['rmse', 'mae', 'precision@10', 'recall@10', 'ndcg@10']) + ' |')
    base_rmse = lookup[('NeuralCF', 'rmse')][0]
    content_rmse = lookup[('ContentAware', 'rmse')][0]
    base_ndcg = lookup[('NeuralCF', 'ndcg@10')][0]
    content_ndcg = lookup[('ContentAware', 'ndcg@10')][0]
    ablation = pd.read_csv(output / 'ablation.csv')
    off = ablation.loc[ablation.variant == 'genre_off'].set_index('seed')
    on = ablation.loc[ablation.variant == 'genre_on'].set_index('seed')
    rmse_wins = int((on.rmse < off.rmse).sum())
    ndcg_wins = int((on['ndcg@10'] > off['ndcg@10']).sum())
    best_rmse = min((model for model, metric in lookup if metric == 'rmse'),
                    key=lambda model: lookup[(model, 'rmse')][0])
    best_ndcg = max((model for model, metric in lookup if metric == 'ndcg@10'),
                    key=lambda model: lookup[(model, 'ndcg@10')][0])
    lines += ['', '## Effect of genre features', '',
              f'With genre features added, RMSE changes by {content_rmse - base_rmse:+.4f} '
              f'(negative = improvement), and NDCG@10 changes by '
              f'{content_ndcg - base_ndcg:+.4f} (positive = improvement).', '',
              f'Across the {len(config["seeds"])} seeds, the genre-aware model beats the baseline '
              f'on RMSE {rmse_wins} times and on NDCG@10 {ndcg_wins} times. '
              f'The lowest mean RMSE among the five algorithms is achieved by {best_rmse}, '
              f'and the highest mean NDCG@10 by {best_ndcg}.', '',
              'These conclusions hold only for this data split, hyperparameters, and training budget. '
              'Means and standard deviations are not significance tests, and other settings '
              'may give different results.', '',
              'Genre off uses the same random seed and batch order, is independently trained from a '
              'freshly initialised model, and does not load baseline weights. '
              'ablation_checks.csv records its differences in predictions, parameters, training loss, '
              'validation performance, and metrics against NeuralCF; '
              'ablation_loss_curves.csv stores the independent training curves, and '
              'execution_log.csv records every real fit.', '',
              '## Sparse-history analysis', '',
              f'Groups use the lowest 25% of training interaction counts: user threshold '
              f'{sparse_info["user_count_threshold"]:g}, movie threshold '
              f'{sparse_info["item_count_threshold"]:g}.', '',
              '| Group | NeuralCF RMSE | Genre-aware RMSE | Test rating rows |',
              '|---|---|---|---|']
    for group, title in [('sparse_user', 'Sparse users'), ('sparse_item', 'Sparse movies')]:
        lines.append(f'| {title} | {value("NeuralCF", group + "_rmse")} | '
                     f'{value("ContentAware", group + "_rmse")} | '
                     f'{sparse_info["test_rating_counts"][group]} |')
    for group, title in [('sparse_user', 'sparse-user'), ('sparse_item', 'sparse-movie')]:
        difference = lookup[('ContentAware', group + '_rmse')][0] - lookup[('NeuralCF', group + '_rmse')][0]
        direction = 'decreases (mean improvement)' if difference < 0 else 'increases (no mean improvement)'
        lines += ['', f'In the {title} group, adding genres changes RMSE by {difference:+.4f}, i.e. {direction}.']
    lines += ['', 'The original sparse-user group contains '
              f'{sparse_info["opposite_side_cold_rows"]["sparse_user"]} cold-movie rows; '
              f'the sparse-movie group contains '
              f'{sparse_info["opposite_side_cold_rows"]["sparse_item"]} cold-user rows, '
              'which use the unified fallback.', '',
              'To isolate the learnable model, we also report sparse subsets where both sides '
              'appear in training:', '',
              '| Group (both sides warm) | NeuralCF RMSE | Genre-aware RMSE | Test rating rows |',
              '|---|---|---|---|']
    for group, title in [('sparse_user_warm', 'Sparse users'), ('sparse_item_warm', 'Sparse movies')]:
        lines.append(f'| {title} | {value("NeuralCF", group + "_rmse")} | '
                     f'{value("ContentAware", group + "_rmse")} | '
                     f'{sparse_info["test_rating_counts"][group]} |')
    lines += ['', 'The two groups may overlap; this is relative sparse-history analysis, '
              'not a zero-interaction cold-start experiment.', '',
              '## Evaluation scope and limitations', '',
              '- RMSE/MAE cover all test ratings; cold users or movies use the unified training-mean fallback. '
              'warm_rmse/warm_mae are also reported in the CSVs for both-sides-warm inspection.',
              '- The global chronological split introduces cold users, so overall rating error can be '
              'strongly affected by the unified fallback. See data_summary.json for counts; not all '
              'error changes can be attributed to genre features.',
              '- Top-K is evaluated only for users seen in training, over the training catalog, '
              'with a positive threshold of 4; full-catalog ranking with no negative sampling. '
              'Movies already seen in training and validation are excluded; users without any '
              'eligible test positive are not included in the macro average.',
              '- Neighbourhood CF is also refit and evaluated per seed. It has no stochastic training, '
              'so its metrics are identical across real runs and its standard deviation is 0.',
              '- All neural models share the same embedding dimension, learning rate, batch size, and '
              'early-stopping rule. The genre-aware model has extra genre parameters, so this ablation '
              'cannot fully separate the effect of content features from added capacity.', '',
              'Figures are in figures/; see EXPERIMENT_PROTOCOL.md in the project root for the full reproduction rules.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def run_experiment(config):
    """Run data preparation, repeated five-model experiments, independent ablations, validation, and figure output."""
    validate_seeds(config)
    started = time.perf_counter()
    torch.set_num_threads(config['threads'])
    output = Path(config['results_dir'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'checkpoints').mkdir(exist_ok=True)
    # Write "running" first; only after all experiments and checks pass do we write "complete".
    write_json(output / 'status.json', {'status': 'running'})
    write_json(output / 'config.json', config)
    write_json(output / 'environment.json', dict(python=platform.python_version(),
               platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
               pandas=pd.__version__, scipy=scipy.__version__, threads=config['threads'], device='cpu'))
    data = load_data(config['data_dir'])
    # Prepare means, ranking candidates, and sparse thresholds once so every model
    # is evaluated on the same scope.
    means = training_means(data)
    ranking = prepare_ranking(data, 'test', config['positive_threshold'])
    masks, sparse_info = sparse_groups(data)
    data_info = summarize(data)
    data_info['ranking_test'] = ranking.metadata
    data_info['sparse_groups'] = sparse_info
    write_json(output / 'data_summary.json', data_info)
    print(f'Data: train={len(data.train)}, val={len(data.val)}, test={len(data.test)}; '
          f'Top-K users={len(ranking.users)}, movies={len(ranking.catalog)}', flush=True)
    rows, histories, ablations, checks, ablation_histories, execution = [], [], [], [], [], []

    def persist():
        """Save current records after each fit so progress is visible and partial failures diagnosable."""
        save_tables(output, rows, histories, ablations, checks, ablation_histories, execution)

    def record_fit(name, seed, role, began, seconds, epochs, best_epoch):
        """Record real fit time and epochs for the final completeness check."""
        execution.append(dict(run_id=f'{name}_seed{seed}', model=name, seed=seed, role=role,
                              started_utc=began, finished_utc=datetime.now(timezone.utc).isoformat(),
                              fit_seconds=seconds, epochs=epochs, best_epoch=best_epoch,
                              independent_fit=True))

    for name, cls in [('UserCF', UserCF), ('ItemCF', ItemCF)]:
        for seed in config['seeds']:
            # Even deterministic algorithms are rebuilt, refit, and re-evaluated per seed.
            set_seed(seed)
            print(f'Fitting {name} seed={seed} independently...', flush=True)
            began = datetime.now(timezone.utc).isoformat()
            fit_start = time.perf_counter()
            model = cls(data.n_users, data.n_items, neighbors=config['neighbors'],
                        shrinkage=config['shrinkage']).fit(data.train)
            seconds = time.perf_counter() - fit_start
            record_fit(name, seed, 'main', began, seconds, 0, 0)
            scores, _ = evaluate_model(model, data, means, ranking, masks, config)
            rows.append(dict(model=name, seed=seed, fit_seconds=seconds, best_epoch=0, **scores))
            print(f'  {name} seed={seed}: RMSE={scores["rmse"]:.4f}, '
                  f'NDCG@10={scores["ndcg@10"]:.4f}', flush=True)
            persist()
    for seed in config['seeds']:
        baseline_model, baseline_scores, baseline_prediction, baseline_history, baseline_epoch = None, None, None, None, None
        for name in ['MF', 'NeuralCF', 'ContentAware']:
            began = datetime.now(timezone.utc).isoformat()
            model, history, best_epoch, seconds = fit_neural(name, data, config, seed, means)
            record_fit(name, seed, 'main', began, seconds, len(history), best_epoch)
            histories.extend(history)
            scores, prediction = evaluate_model(model, data, means, ranking, masks, config)
            rows.append(dict(model=name, seed=seed, fit_seconds=seconds, best_epoch=best_epoch, **scores))
            print(f'  TEST {name} seed={seed}: RMSE={scores["rmse"]:.4f}, '
                  f'NDCG@10={scores["ndcg@10"]:.4f}', flush=True)
            # Save only the first seed's weights for demo recommendations;
            # remaining seeds are still fully trained and evaluated.
            if seed == config['seeds'][0]:
                torch.save(dict(model=name, seed=seed, config=config, state_dict=model.state_dict()),
                           output / 'checkpoints' / f'{name}_seed{seed}.pt')
            if name == 'NeuralCF':
                baseline_model, baseline_scores = model, scores
                baseline_prediction, baseline_history, baseline_epoch = prediction, history, best_epoch
            if name == 'ContentAware':
                ablations.append(dict(seed=seed, variant='genre_on', **scores,
                                      max_abs_difference=float(np.max(np.abs(prediction - baseline_prediction)))))
            persist()

        # Ablation trains a fresh model independently; loading baseline weights would not
        # count as reproduction.
        began = datetime.now(timezone.utc).isoformat()
        disabled, off_history, off_epoch, off_seconds = fit_neural(
            'GenreOff', data, config, seed, means, use_genres=False)
        record_fit('GenreOff', seed, 'ablation_off', began, off_seconds, len(off_history), off_epoch)
        off_scores, off_prediction = evaluate_model(disabled, data, means, ranking, masks, config)
        ablation_histories.extend(off_history)
        if len(off_history) != len(baseline_history):
            raise ValueError('Independent ablation has different training epochs')
        baseline_state, off_state = baseline_model.state_dict(), disabled.state_dict()
        if set(baseline_state) != set(off_state):
            raise ValueError('Genre-off parameter structure differs from NeuralCF')
        # Compare parameters, predictions, and per-epoch curves to confirm the
        # genre-off branch independently reproduces the baseline.
        parameter_difference = max(float((value.double() - off_state[key].double()).abs().max())
                                   for key, value in baseline_state.items())
        prediction_difference = float(np.max(np.abs(off_prediction - baseline_prediction)))
        check = dict(seed=seed, best_epoch=off_epoch, baseline_best_epoch=baseline_epoch,
                     independently_trained=True, max_abs_prediction_difference=prediction_difference,
                     max_abs_parameter_difference=parameter_difference,
                     max_abs_train_loss_difference=max(abs(a['train_mse'] - b['train_mse'])
                         for a, b in zip(off_history, baseline_history)),
                     max_abs_val_rmse_difference=max(abs(a['val_rmse'] - b['val_rmse'])
                         for a, b in zip(off_history, baseline_history)),
                     max_abs_metric_difference=max(abs(off_scores[key] - baseline_scores[key])
                         for key in required_metrics(config)))
        checks.append(check)
        ablations.append(dict(seed=seed, variant='genre_off', **off_scores,
                              max_abs_difference=prediction_difference))
        if seed == config['seeds'][0]:
            torch.save(dict(model='GenreOff', seed=seed, config=config, state_dict=off_state),
                       output / 'checkpoints' / f'GenreOff_seed{seed}.pt')
        print(f'  INDEPENDENT GenreOff seed={seed}: max prediction difference='
              f'{prediction_difference:.3g}', flush=True)
        persist()

    # Only after all records are present, metrics are valid, and the independent
    # ablation matches, do we write the official completion marker.
    try:
        validation = validate_results(pd.DataFrame(rows), pd.DataFrame(ablations),
                                      pd.DataFrame(checks), pd.DataFrame(execution), config)
    except ValueError as error:
        write_json(output / 'status.json', dict(status='failed', error=str(error)))
        raise
    write_json(output / 'validation.json', validation)
    plot_all(output)
    elapsed = time.perf_counter() - started
    write_report(output, config, data_info, sparse_info, elapsed)
    write_json(output / 'status.json', dict(status='complete', elapsed_seconds=elapsed,
               model_seed_rows=len(rows), independent_fits=len(execution),
               ablation_rows=len(ablations), seeds=config['seeds'], sampled_data=False,
               validation='passed', formal=validation['formal']))
    print(f'Finished in {elapsed / 60:.1f} minutes. Results: {output.resolve()}', flush=True)