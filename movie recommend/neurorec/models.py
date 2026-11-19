"""The five rating predictors in the assignment: UserCF, ItemCF, MF, and two NeuralCF variants with a genre switch.

The two neighbourhood CF models first remove per-entity mean rating differences, then
look for positively similar neighbours; similarity is shrunk by co-rating counts to
reduce the effect of accidental overlap. Ratings and neighbours are stored as sparse
matrices, suitable for MovieLens 1M.
Neural models return raw scores here; the evaluation layer handles rating-range clipping
uniformly, while ranking keeps the raw scores.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import sparse
import torch
from torch import nn


def _training_matrices(train, n_users: int, n_items: int):
    """Build the sparse rating matrix, observed mask, and global mean from training ratings only."""
    users = train["user"].to_numpy(dtype=np.int64)
    items = train["item"].to_numpy(dtype=np.int64)
    ratings = train["rating"].to_numpy(dtype=np.float32)
    if len(ratings) == 0:
        raise ValueError("The training set is empty.")
    if not np.isfinite(ratings).all():
        raise ValueError("Training ratings must be finite.")
    if users.min() < 0 or users.max() >= n_users or items.min() < 0 or items.max() >= n_items:
        raise ValueError("User/item index is outside the configured model size.")
    shape = (n_users, n_items)
    observed = sparse.csr_matrix((np.ones(len(ratings), np.float32), (users, items)), shape=shape)
    if observed.max() > 1:
        raise ValueError("Duplicate user-item training ratings must be resolved before fitting.")
    matrix = sparse.csr_matrix((ratings, (users, items)), shape=shape)
    return matrix, observed, float(ratings.mean())


def _entity_means(matrix, observed, global_mean: float, axis: int) -> np.ndarray:
    """Row- or column-wise mean; entities without training records fall back to the global mean to avoid division by zero."""
    counts = np.asarray(observed.sum(axis=axis)).ravel()
    sums = np.asarray(matrix.sum(axis=axis)).ravel()
    means = np.full(len(counts), global_mean, dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means


def _topk_similarities(
    centred: sparse.csr_matrix,
    observed: sparse.csr_matrix,
    neighbors: int,
    shrinkage: float,
    block_size: int = 128,
) -> sparse.csr_matrix:
    """Compute per-entity neighbours in blocks so the full similarity matrix is never materialised.

    Each row of `centred` is an entity's residual rating. Keep only the `neighbors`
    strongest positive similarities. Co-rating counts must come from `observed`:
    a rating equal to the mean is still a real record.
    """
    n_entities = centred.shape[0]
    # Normalise each user's (or movie's) residual vector so the dot product equals cosine similarity.
    norm = np.sqrt(np.asarray(centred.multiply(centred).sum(axis=1)).ravel())
    inv_norm = np.zeros_like(norm)
    np.divide(1.0, norm, out=inv_norm, where=norm > 0)
    normalised = (sparse.diags(inv_norm) @ centred).tocsr()
    transpose = normalised.T.tocsc()
    observed_transpose = observed.T.tocsc()
    all_indices, all_values = [], []
    indptr = [0]
    for start in range(0, n_entities, block_size):
        end = min(start + block_size, n_entities)
        sim = (normalised[start:end] @ transpose).tocsr()
        if shrinkage > 0:
            # The fewer the co-ratings, the less reliable the similarity; down-weight it
            # by the shrinkage factor.
            common = (observed[start:end] @ observed_transpose).tocsr()
            common.data /= common.data + shrinkage
            sim = sim.multiply(common).tocsr()
        for row in range(end - start):
            begin, finish = sim.indptr[row], sim.indptr[row + 1]
            indices = sim.indices[begin:finish]
            values = sim.data[begin:finish]
            # Remove self and negative similarities; keep the strongest remaining neighbours.
            keep = (indices != start + row) & (values > 0) & np.isfinite(values)
            indices, values = indices[keep], values[keep]
            if len(values) > neighbors:
                # Break similarity ties by index for a deterministic neighbour set across runs.
                order = np.lexsort((indices, -values))[:neighbors]
                indices, values = indices[order], values[order]
            order = np.argsort(indices)
            all_indices.extend(indices[order].tolist())
            all_values.extend(values[order].tolist())
            indptr.append(len(all_indices))
    return sparse.csr_matrix(
        (np.asarray(all_values, np.float32), np.asarray(all_indices, np.int32), np.asarray(indptr, np.int64)),
        shape=(n_entities, n_entities),
    )


class _NeighbourBase:
    """Shared parameter checks and batched predict API for both neighbourhood methods; indices are preprocessed contiguous IDs."""

    def __init__(self, n_users: int, n_items: int, neighbors: int = 40, shrinkage: float = 10.0):
        if n_users < 1 or n_items < 1 or neighbors < 1 or shrinkage < 0:
            raise ValueError("Invalid neighbourhood model configuration.")
        self.n_users, self.n_items = int(n_users), int(n_items)
        self.neighbors, self.shrinkage = int(neighbors), float(shrinkage)
        self.is_fitted = False

    def _check_user_items(self, user: int, items) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Fit the model before predicting.")
        items = np.asarray(items, dtype=np.int64).reshape(-1)
        if not 0 <= user < self.n_users or np.any(items < 0) or np.any(items >= self.n_items):
            raise ValueError("User/item index is outside the configured model size.")
        return items

    def predict(self, users, items) -> np.ndarray:
        """Predict a batch of user-movie pairs; group by user internally, preserving output order."""
        users = np.asarray(users, dtype=np.int64).reshape(-1)
        items = np.asarray(items, dtype=np.int64).reshape(-1)
        if len(users) != len(items):
            raise ValueError("users and items must have the same length.")
        result = np.empty(len(users), dtype=np.float32)
        # Sort to obtain contiguous user groups; avoids rescanning the full test set per user.
        order = np.argsort(users, kind="stable")
        ordered_users = users[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_users)) + 1, len(users)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start == end:
                continue
            positions = order[start:end]
            result[positions] = self.score_items(int(ordered_users[start]), items[positions])
        return result


class UserCF(_NeighbourBase):
    """User-based CF: similar users' ratings of a movie inform the target user's prediction."""

    def fit(self, train):
        """Learn user means and neighbours from the training set only; val/test ratings never enter the similarity."""
        matrix, self.observed, self.global_mean = _training_matrices(train, self.n_users, self.n_items)
        self.user_means = _entity_means(matrix, self.observed, self.global_mean, axis=1)
        self.item_means = _entity_means(matrix, self.observed, self.global_mean, axis=0)
        self.centred = matrix.copy()
        row_indices = np.repeat(np.arange(self.n_users), np.diff(matrix.indptr))
        # Subtract user means to remove individual rating-scale differences.
        self.centred.data -= self.user_means[row_indices]
        self.centred.eliminate_zeros()
        self.similarity = _topk_similarities(self.centred, self.observed, self.neighbors, self.shrinkage)
        self.is_fitted = True
        return self

    def score_items(self, user: int, items) -> np.ndarray:
        """Predict several movies for one user; fall back to the user's training mean if no neighbour ratings are available."""
        items = self._check_user_items(user, items)
        weights = self.similarity.getrow(user)
        # Weight similar users' residuals by similarity, then add the target user's mean back.
        numerator = (weights @ self.centred).toarray().ravel()[items]
        denominator = (weights @ self.observed).toarray().ravel()[items]
        residual = np.zeros(len(items), dtype=np.float32)
        np.divide(numerator, denominator, out=residual, where=denominator > 1e-12)
        return self.user_means[user] + residual


class ItemCF(_NeighbourBase):
    """Item-based CF: predict a target movie using a user's ratings of similar movies."""

    def fit(self, train):
        """Learn item means and neighbours from the training set only; transpose so each row is a movie."""
        matrix, self.observed, self.global_mean = _training_matrices(train, self.n_users, self.n_items)
        self.user_means = _entity_means(matrix, self.observed, self.global_mean, axis=1)
        self.item_means = _entity_means(matrix, self.observed, self.global_mean, axis=0)
        self.centred = matrix.copy()
        # Movies are the similarity entities here, so subtract per-movie means.
        self.centred.data -= self.item_means[self.centred.indices]
        self.centred.eliminate_zeros()
        self.similarity = _topk_similarities(
            self.centred.T.tocsr(), self.observed.T.tocsr(), self.neighbors, self.shrinkage
        )
        self.is_fitted = True
        return self

    def score_items(self, user: int, items) -> np.ndarray:
        """Weighted residual prediction using each target movie's neighbours; fall back to the item mean when unavailable."""
        items = self._check_user_items(user, items)
        weights = self.similarity[items]
        # Predict the target movie from the user's residuals on similar movies.
        numerator = (weights @ self.centred.getrow(user).T).toarray().ravel()
        denominator = (weights @ self.observed.getrow(user).T).toarray().ravel()
        residual = np.zeros(len(items), dtype=np.float32)
        np.divide(numerator, denominator, out=residual, where=denominator > 1e-12)
        return self.item_means[items] + residual


class MatrixFactorization(nn.Module):
    """Matrix factorisation with biases: latent dot product captures preference; biases capture per-entity mean rating gaps."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 32, global_mean: float = 3.5):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.register_buffer("global_mean", torch.tensor(float(global_mean), dtype=torch.float32))
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Prediction = global mean + user bias + item bias + dot product of two latent vectors."""
        interaction = (self.user_embedding(users) * self.item_embedding(items)).sum(dim=-1)
        bias = self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1)
        return self.global_mean + bias + interaction


class NeuralCF(nn.Module):
    """A single MLP-style NeuralCF implementation; `use_genres` toggles genre information.

    The baseline concatenates user and movie vectors into an MLP for nonlinear
    preference modelling; means and biases provide a rating baseline. The improved
    model averages the movie's genre vectors, projects them, and adds them to the
    movie vector. Additive fusion keeps the MLP input width identical across the
    two variants, which simplifies controlled comparison.
    With genres disabled, no extra layers are created and no extra random numbers are
    consumed, so independent training can reproduce the baseline.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genres: int,
        genre_ids,
        embedding_dim: int = 32,
        hidden_dims: Sequence[int] = (64, 32),
        use_genres: bool = False,
        global_mean: float = 3.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embedding_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid neural model dimensions/dropout.")
        genre_tensor = torch.as_tensor(np.asarray(genre_ids), dtype=torch.long)
        if genre_tensor.ndim != 2 or genre_tensor.shape[0] != n_items:
            raise ValueError("genre_ids must have shape [n_items, max_genres].")
        if genre_tensor.numel() and (genre_tensor.min() < 0 or genre_tensor.max() >= n_genres):
            raise ValueError("Genre ID is outside [0, n_genres).")
        self.use_genres = bool(use_genres)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        layers = []
        width = 2 * embedding_dim
        for hidden in hidden_dims:
            if hidden < 1:
                raise ValueError("Hidden dimensions must be positive.")
            layers.extend([nn.Linear(width, int(hidden)), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = int(hidden)
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)
        self.register_buffer("global_mean", torch.tensor(float(global_mean), dtype=torch.float32))
        self.register_buffer("genre_ids", genre_tensor)
        # Initialise all baseline parameters first, then create the genre layers,
        # so the baseline portion starts from identical values under the same seed.
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)
        if self.use_genres:
            self.genre_embedding = nn.Embedding(n_genres, embedding_dim, padding_idx=0)
            self.genre_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
            nn.init.normal_(self.genre_embedding.weight, std=0.05)
            with torch.no_grad():
                self.genre_embedding.weight[0].zero_()
                nn.init.eye_(self.genre_projection.weight)
        else:
            self.genre_embedding = None
            self.genre_projection = None

    def mean_genre_embedding(self, items: torch.Tensor) -> torch.Tensor:
        """Average the real genre vectors of each movie; padding and missing genres are excluded from the denominator."""
        if not self.use_genres:
            raise RuntimeError("Genre features are disabled.")
        ids = self.genre_ids[items]
        # A movie can have multiple genres; average only real ones and ignore the padded 0.
        mask = (ids != 0).unsqueeze(-1)
        vectors = self.genre_embedding(ids) * mask
        denominator = mask.sum(dim=-2).clamp_min(1)
        return vectors.sum(dim=-2) / denominator

    def item_features(self, items: torch.Tensor) -> torch.Tensor:
        """Return the movie vector; when enabled, fuse genre content that is trained jointly."""
        features = self.item_embedding(items)
        if self.use_genres:
            # Add genre information to the movie vector; disabling the switch restores plain NeuralCF.
            features = features + self.genre_projection(self.mean_genre_embedding(items))
        return features

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Concatenate user and fused movie vectors, produce a residual through the MLP, then add means and biases."""
        user = self.user_embedding(users)
        item = self.item_features(items)
        neural = self.output(self.mlp(torch.cat([user, item], dim=-1))).squeeze(-1)
        biases = self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1)
        return self.global_mean + biases + neural