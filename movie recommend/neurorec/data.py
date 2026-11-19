"""Load and clean MovieLens 1M, and split it in global chronological order.

Movie genres and ID mappings are static metadata and may be read from the full data;
interaction counts and other statistics use only the training split.
Cold users and cold movies from the rating task are preserved after the temporal split
and predicted by the upper layers via a unified fallback, so untrained embeddings are
never treated as reliable outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import shutil
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
# Checksum of the original GroupLens archive; also listed in the TensorFlow Datasets download table.
ARCHIVE_SIZE = 5_917_549
ARCHIVE_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"


@dataclass
class DataBundle:
    """Data, index mappings, and training statistics shared by all models.

    The user/item columns in the rating tables are 0-based contiguous indices;
    user_ids/item_ids keep the original IDs.
    Rows of genre_ids correspond to movie indices, 0 marks padding; training
    interaction counts are used to identify cold-start samples.
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    genre_ids: np.ndarray
    n_genres: int
    movies: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray
    train_user_counts: np.ndarray
    train_item_counts: np.ndarray
    genre_names: list[str]
    metadata: dict[str, Any]


def _validate_archive(archive: Path) -> None:
    """Verify file size and hash before extraction; reject incomplete or tampered archives."""
    if archive.stat().st_size != ARCHIVE_SIZE:
        raise ValueError(
            f"MovieLens ZIP size mismatch: expected {ARCHIVE_SIZE} bytes, "
            f"got {archive.stat().st_size}. Replace {archive} with a fresh download from {DATA_URL}."
        )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError(
            f"MovieLens ZIP SHA-256 mismatch. Replace {archive} with a fresh download from {DATA_URL}."
        )


def _extract_safely(archive: Path, destination: Path) -> None:
    """Inspect archive member paths to prevent writing outside the target directory."""
    base = destination.resolve()
    with ZipFile(archive) as z:
        for info in z.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
                raise ValueError(f"Unsafe archive member: {info.filename}")
            target = (destination / info.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Archive member escapes data directory: {info.filename}")
        z.extractall(destination)


def _find_or_download(data_dir: Path) -> Path:
    """Prefer existing raw files; otherwise validate the local archive or download from the official URL."""
    if (data_dir / "ratings.dat").is_file() and (data_dir / "movies.dat").is_file():
        return data_dir
    raw = data_dir / "ml-1m"
    if (raw / "ratings.dat").is_file() and (raw / "movies.dat").is_file():
        return raw
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "ml-1m.zip"
    if not archive.is_file():
        partial = data_dir / "ml-1m.zip.part"
        request = Request(DATA_URL, headers={"User-Agent": "ContentAwareNeuroRec/1.0"})
        print(f"Downloading official MovieLens 1M: {DATA_URL}", flush=True)
        try:
            with urlopen(request, timeout=120) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            partial.replace(archive)
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                "Could not securely download MovieLens 1M. Check your network and Python's "
                "trusted certificates, or download " + DATA_URL + " using your browser and "
                "save it as " + str(archive.resolve()) + ". Then run the same command again. "
                "TLS certificate verification has not been disabled."
            ) from exc
        finally:
            if partial.exists():
                partial.unlink()
    _validate_archive(archive)
    _extract_safely(archive, data_dir)
    if not (raw / "ratings.dat").is_file() or not (raw / "movies.dat").is_file():
        raise ValueError("MovieLens ZIP does not contain ml-1m/ratings.dat and movies.dat")
    return raw


def _time_boundary(timestamps: np.ndarray, fraction: float) -> int:
    """Move the cut point to the first record with the same timestamp so ties do not cross splits."""
    desired = min(max(int(len(timestamps) * fraction), 1), len(timestamps) - 1)
    return int(np.searchsorted(timestamps, timestamps[desired], side="left"))


def load_data(
    data_dir: str | Path, train_fraction: float = 0.8, val_fraction: float = 0.1
) -> DataBundle:
    """Clean, encode features, and split by time; default train/val/test is roughly 80/10/10.

    A global chronological split makes val and test strictly later than train,
    preventing future interactions from leaking into training.
    All models share this fixed split; random seeds only affect subsequent training.
    """
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("train_fraction and val_fraction must be positive and sum to less than 1")
    raw = _find_or_download(Path(data_dir))
    ratings = pd.read_csv(
        raw / "ratings.dat", sep="::", engine="python", encoding="latin-1",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        raw / "movies.dat", sep="::", engine="python", encoding="latin-1",
        names=["movie_id", "title", "genres"],
    )
    initial_rows = len(ratings)
    required = ["user_id", "movie_id", "rating", "timestamp"]
    for column in required:
        ratings[column] = pd.to_numeric(ratings[column], errors="coerce")
    valid = ratings[required].notna().all(axis=1) & ratings.rating.between(1, 5)
    valid &= (ratings.user_id > 0) & (ratings.movie_id > 0) & (ratings.timestamp >= 0)
    valid &= (ratings.user_id % 1 == 0) & (ratings.movie_id % 1 == 0) & (ratings.timestamp % 1 == 0)
    invalid_rows = int((~valid).sum())
    ratings = ratings.loc[valid].copy()
    for column in ["user_id", "movie_id", "timestamp"]:
        ratings[column] = ratings[column].astype(np.int64)
    movies = movies.dropna(subset=["movie_id", "title", "genres"]).copy()
    movies.movie_id = movies.movie_id.astype(np.int64)
    if movies.movie_id.duplicated().any():
        raise ValueError("movies.dat contains duplicate movie IDs")
    unknown = ~ratings.movie_id.isin(movies.movie_id)
    if unknown.any():
        raise ValueError(f"{int(unknown.sum())} ratings refer to missing movie metadata")
    ratings = ratings.sort_values(["timestamp", "user_id", "movie_id"], kind="stable")
    # Keep the earliest rating so future duplicates cannot decide whether an early training row exists;
    # original ML-1M normally has no duplicates.
    duplicate_rows = int(ratings.duplicated(["user_id", "movie_id"], keep="first").sum())
    ratings = ratings.drop_duplicates(["user_id", "movie_id"], keep="first").reset_index(drop=True)
    if len(ratings) < 3:
        raise ValueError("At least three valid, distinct rating rows are required")
    # Original IDs need not be contiguous; remapping lets us index embeddings directly.
    # The mapping itself carries no learnable rating information.
    user_ids = np.sort(ratings.user_id.unique()).astype(np.int64)
    movies = movies.sort_values("movie_id").reset_index(drop=True)
    item_ids = movies.movie_id.to_numpy(dtype=np.int64)
    user_map = {int(value): index for index, value in enumerate(user_ids)}
    item_map = {int(value): index for index, value in enumerate(item_ids)}
    frame = pd.DataFrame({
        "user": ratings.user_id.map(user_map).to_numpy(dtype=np.int64),
        "item": ratings.movie_id.map(item_map).to_numpy(dtype=np.int64),
        "rating": ratings.rating.to_numpy(dtype=np.float32),
        "timestamp": ratings.timestamp.to_numpy(dtype=np.int64),
    })
    times = frame.timestamp.to_numpy()
    cut1 = _time_boundary(times, train_fraction)
    cut2 = _time_boundary(times, train_fraction + val_fraction)
    if not 0 < cut1 < cut2 < len(frame):
        raise ValueError("Timestamp ties leave an empty split; choose different fractions or provide more time points")
    train = frame.iloc[:cut1].reset_index(drop=True)
    val = frame.iloc[cut1:cut2].reset_index(drop=True)
    test = frame.iloc[cut2:].reset_index(drop=True)
    movies.insert(0, "item", np.arange(len(movies), dtype=np.int64))
    # Movies have different numbers of genres; pad with 0 and ignore padding during pooling
    # so real genre vectors are not diluted.
    genre_names = ["<PAD>"] + sorted({g for text in movies.genres for g in str(text).split("|") if g})
    genre_map = {value: index for index, value in enumerate(genre_names)}
    lists = [[genre_map[g] for g in str(value).split("|") if g] for value in movies.genres]
    genre_ids = np.zeros((len(movies), max(map(len, lists), default=1)), dtype=np.int64)
    for index, ids in enumerate(lists):
        genre_ids[index, :len(ids)] = ids
    # Cold-start detection and sparse grouping rely only on training history;
    # val/test interaction counts must not be borrowed.
    train_user_counts = np.bincount(train.user, minlength=len(user_ids)).astype(np.int64)
    train_item_counts = np.bincount(train.item, minlength=len(item_ids)).astype(np.int64)
    archive = Path(data_dir) / "ml-1m.zip"
    metadata = {
        "source_url": DATA_URL,
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest() if archive.is_file() else None,
        "split": "global_chronological",
        "requested_train_fraction": train_fraction,
        "requested_val_fraction": val_fraction,
        "timestamp_ties": "kept in the same partition",
        "raw_rating_rows": initial_rows,
        "invalid_rating_rows_removed": invalid_rows,
        "duplicate_user_item_rows_removed": duplicate_rows,
        "duplicate_policy": "retain earliest timestamp for each user-item pair",
        "clean_rating_rows": len(frame),
        "id_mapping": "sorted original IDs; mappings and movie genres are static metadata only",
    }
    return DataBundle(
        train, val, test, len(user_ids), len(item_ids), genre_ids, len(genre_names),
        movies, user_ids, item_ids, train_user_counts, train_item_counts, genre_names, metadata,
    )


def summarize(bundle: DataBundle) -> dict[str, Any]:
    """Summarise actual split sizes, time range, and cold-start counts to clarify metric coverage."""
    result = dict(bundle.metadata)
    result.update({
        "n_users": bundle.n_users, "n_items": bundle.n_items,
        "n_genres_excluding_padding": bundle.n_genres - 1,
        "train_catalog_size": int((bundle.train_item_counts > 0).sum()),
        "train_user_count": int((bundle.train_user_counts > 0).sum()),
        "partitions": {},
    })
    for name in ("train", "val", "test"):
        frame = getattr(bundle, name)
        cold_user = bundle.train_user_counts[frame.user.to_numpy()] == 0
        cold_item = bundle.train_item_counts[frame.item.to_numpy()] == 0
        result["partitions"][name] = {
            "rows": len(frame), "users": int(frame.user.nunique()), "items": int(frame.item.nunique()),
            "min_timestamp": int(frame.timestamp.min()), "max_timestamp": int(frame.timestamp.max()),
            "min_utc": datetime.fromtimestamp(int(frame.timestamp.min()), timezone.utc).isoformat(),
            "max_utc": datetime.fromtimestamp(int(frame.timestamp.max()), timezone.utc).isoformat(),
            "cold_user_rows": int(cold_user.sum()), "cold_item_rows": int(cold_item.sum()),
            "warm_rows": int((~cold_user & ~cold_item).sum()),
        }
    return result