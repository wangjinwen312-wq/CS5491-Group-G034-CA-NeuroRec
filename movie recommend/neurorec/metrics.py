"""Compute rating error under unified evaluation rules and prepare Top-K ranking data with binary relevance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data import DataBundle


def evaluate_rating(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE and MAE; reject empty arrays or non-finite values so no invalid result is produced."""
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if truth.shape != prediction.shape or not len(truth):
        raise ValueError("Rating arrays must have the same nonzero length")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Rating arrays must contain finite values")
    error = prediction - truth
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))}


def ranking_metrics(
    recommended: np.ndarray | Iterable[int], relevant: set[int], ks: Iterable[int] = (5, 10, 20)
) -> dict[str, float]:
    """Compute Precision@K, Recall@K, and binary NDCG@K for one user.

    `recommended` is sorted by predicted score by the caller; `relevant` is the
    user's held-out positive set. Precision divides by K (short lists contribute
    misses), Recall divides by the number of positives.
    Users without positives are filtered out during preparation; the caller
    averages per-user metrics with equal weights (macro average).
    """
    rec = np.asarray(list(recommended), dtype=np.int64).reshape(-1)
    if len(np.unique(rec)) != len(rec):
        raise ValueError("A recommendation list must not contain duplicate items")
    if not relevant:
        raise ValueError("Ranking metrics require at least one relevant held-out item")
    # Use binary relevance: membership in the positive set counts as 1;
    # raw star ratings do not re-weight hits.
    hits = np.asarray([int(item) in relevant for item in rec], dtype=np.float64)
    result: dict[str, float] = {}
    for k in ks:
        k = int(k)
        if k < 1:
            raise ValueError("K must be positive")
        prefix = hits[:k]
        discount = 1.0 / np.log2(np.arange(2, len(prefix) + 2))
        dcg = float(np.sum(prefix * discount))
        # The ideal ranking contains at most min(K, |relevant|) hits; use it to
        # normalise DCG across users.
        ideal_length = min(k, len(relevant))
        idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_length + 2))))
        result[f"precision@{k}"] = float(prefix.sum() / k)
        result[f"recall@{k}"] = float(prefix.sum() / len(relevant))
        result[f"ndcg@{k}"] = dcg / idcg
    return result


@dataclass
class RankingData:
    """Unified ranking evaluation inputs: eligible users, catalog, positives, seen history, scope info."""
    users: np.ndarray
    catalog: np.ndarray
    relevant: dict[int, set[int]]
    seen: dict[int, set[int]]
    metadata: dict[str, Any]


def prepare_ranking(bundle: DataBundle, split: str = "test", threshold: float = 4.0) -> RankingData:
    """Prepare ranking evaluation over users seen in training and the full training catalog; no negative sampling.

    Validation excludes training history; test excludes train and validation history,
    preventing recommendation of items already seen at evaluation time.
    Positives are read only from the specified held-out split; every model uses the
    same users, labels, and candidate filtering for comparability.
    """
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    if not 1 <= threshold <= 5:
        raise ValueError("The positive rating threshold must lie in [1, 5]")
    heldout = getattr(bundle, split)
    history = bundle.train if split == "val" else pd.concat([bundle.train, bundle.val], ignore_index=True)
    seen = {int(user): set(group.item.to_numpy(dtype=np.int64).tolist())
            for user, group in history.groupby("user", sort=True)}
    # The catalog only contains movies that appear in training, so future interactions
    # cannot decide the catalog and untrained movie vectors are never used.
    catalog = np.flatnonzero(bundle.train_item_counts > 0).astype(np.int64)
    warm_user_mask = bundle.train_user_counts[heldout.user.to_numpy()] > 0
    warm_item_mask = bundle.train_item_counts[heldout.item.to_numpy()] > 0
    # Positives must reach the like threshold, and both user and movie must appear
    # in training so the target is covered by the same candidate rule.
    positives = heldout.loc[(heldout.rating >= threshold) & warm_user_mask & warm_item_mask]
    relevant: dict[int, set[int]] = {}
    for user, group in positives.groupby("user", sort=True):
        user = int(user)
        targets = set(group.item.to_numpy(dtype=np.int64).tolist()) - seen.get(user, set())
        if targets:
            relevant[user] = targets
    # Recall is undefined without an eligible positive; drop those users and report
    # the count in metadata so the denominator is transparent.
    users = np.asarray(sorted(relevant), dtype=np.int64)
    heldout_warm_users = set(heldout.loc[warm_user_mask, "user"].astype(int).tolist())
    metadata = {
        "split": split, "positive_threshold": threshold,
        "catalog": "movies with at least one training interaction",
        "catalog_size": len(catalog), "candidate_sampling": "none: full eligible catalog",
        "history_excluded": "train" if split == "val" else "train + validation",
        "heldout_users": int(heldout.user.nunique()), "warm_heldout_users": len(heldout_warm_users),
        "evaluated_users": len(users),
        "cold_users_excluded": int(heldout.user.nunique()) - len(heldout_warm_users),
        "warm_users_without_eligible_positives_excluded": len(heldout_warm_users) - len(users),
        "heldout_positive_rows": int((heldout.rating >= threshold).sum()),
        "eligible_positive_rows": int(sum(map(len, relevant.values()))),
        "no_positive_users": "excluded from macro average; exclusion counts reported",
        "relevance": "binary; observed held-out rating >= threshold; unobserved is unknown",
        "averaging": "unweighted mean across eligible users",
    }
    return RankingData(users, catalog, relevant, seen, metadata)