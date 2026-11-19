"""Unified training and prediction: learn from training ratings, early-stop on validation, never select on test.

Neural models fit explicit ratings by MSE; every fit builds a fresh model and
optimiser, so independent ablation can reproduce the baseline.
The prediction interface also supports classical CF and uniformly applies rating
clipping and cold-start fallback for users/movies unseen in training.
"""
from copy import deepcopy
import random
import time

import numpy as np
import torch

from .metrics import evaluate_rating
from .models import MatrixFactorization, NeuralCF


def set_seed(seed):
    """Fix initialisation-related randomness and require deterministic PyTorch ops for cross-run verification."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_model(name, data, config, use_genres=None):
    """Construct a new model from config; an explicit genre switch overrides the name, used for independent GenreOff training."""
    common = dict(n_users=data.n_users, n_items=data.n_items,
                  embedding_dim=config['embedding_dim'],
                  global_mean=float(data.train.rating.mean()))
    if name == 'MF':
        return MatrixFactorization(**common)
    return NeuralCF(**common, n_genres=data.n_genres, genre_ids=data.genre_ids,
                    hidden_dims=config['hidden_dims'],
                    use_genres=(name == 'ContentAware') if use_genres is None else use_genres)


def raw_predict(model, users, items, batch_size=65536):
    """Unified batched predict interface returning unclipped scores, so ranking is not affected by clipping-induced ties."""
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    if not isinstance(model, torch.nn.Module):
        return model.predict(users, items).astype(np.float64)
    model.eval()
    result = []
    # No gradients for inference; batching limits peak memory when scoring the full catalog.
    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            u = torch.from_numpy(users[start:start + batch_size])
            i = torch.from_numpy(items[start:start + batch_size])
            result.append(model(u, i).numpy())
    return np.concatenate(result).astype(np.float64) if result else np.array([])


def training_means(data):
    """Compute global, user, and item means from the training set only, providing a consistent fallback for all algorithms."""
    global_mean = float(data.train.rating.mean())
    user_means = np.full(data.n_users, global_mean)
    item_means = np.full(data.n_items, global_mean)
    u = data.train.groupby('user').rating.mean()
    i = data.train.groupby('item').rating.mean()
    user_means[u.index] = u.to_numpy()
    item_means[i.index] = i.to_numpy()
    return global_mean, user_means, item_means


def rating_predict(model, users, items, data, means):
    """Apply the same cold-start fallback across algorithms and clip ratings to [1, 5]."""
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    warm_u = data.train_user_counts[users] > 0
    warm_i = data.train_item_counts[items] > 0
    warm = warm_u & warm_i
    # Fresh embeddings for unseen IDs have learned no preference; do not treat
    # random initial values as reliable ratings.
    global_mean, user_means, item_means = means
    prediction = np.full(len(users), global_mean)
    prediction[warm_u & ~warm_i] = user_means[users[warm_u & ~warm_i]]
    prediction[~warm_u & warm_i] = item_means[items[~warm_u & warm_i]]
    if warm.any():
        prediction[warm] = raw_predict(model, users[warm], items[warm])
    return np.clip(prediction, 1, 5)


def fit_neural(name, data, config, seed, means, use_genres=None):
    """Train from random initialisation; return the best model, per-epoch log, best epoch, and training time.

    Explicitly disabling genres trains the ablation independently; no other
    model's weights are accepted or loaded.
    The objective is MSE against real ratings; validation RMSE is used only for
    early stopping and to choose the best weights of this fit.
    """
    set_seed(seed)
    model = make_model(name, data, config, use_genres=use_genres)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    users = torch.tensor(data.train.user.to_numpy(), dtype=torch.long)
    items = torch.tensor(data.train.item.to_numpy(), dtype=torch.long)
    ratings = torch.tensor(data.train.rating.to_numpy(), dtype=torch.float32)
    # A separate generator keeps the training sample order identical even when
    # genre layers are added.
    generator = torch.Generator().manual_seed(seed)
    best_rmse, stale = float('inf'), 0
    best_state, best_epoch = None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        order = torch.randperm(len(ratings), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), config['batch_size']):
            batch = order[start:start + config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(users[batch], items[batch])
            loss = torch.nn.functional.mse_loss(prediction, ratings[batch])
            if not torch.isfinite(loss):
                raise RuntimeError(f'{name}: non-finite training loss')
            loss.backward()
            optimizer.step()
            # The last batch may be smaller; weight by sample count to obtain the true epoch mean loss.
            total_loss += loss.item() * len(batch)
        prediction = rating_predict(model, data.val.user, data.val.item, data, means)
        scores = evaluate_rating(data.val.rating.to_numpy(), prediction)
        history.append(dict(model=name, seed=seed, epoch=epoch,
                            train_mse=total_loss / len(ratings),
                            val_rmse=scores['rmse'], val_mae=scores['mae']))
        print(f'  {name} seed={seed} epoch={epoch:02d} '
              f'train_MSE={history[-1]["train_mse"]:.4f} val_RMSE={scores["rmse"]:.4f}', flush=True)
        if scores['rmse'] < best_rmse - 1e-6:
            best_rmse = scores['rmse']
            # A deep copy is required; otherwise saved tensors keep updating and the
            # best epoch cannot be truly restored.
            best_epoch, best_state, stale = epoch, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config['patience']:
                break
    # Restore only this fit's own best state; the baseline weights must not be
    # substituted for the independent ablation training.
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch, time.perf_counter() - started