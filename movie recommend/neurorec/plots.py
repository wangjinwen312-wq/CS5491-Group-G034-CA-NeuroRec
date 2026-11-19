"""Read real experiment records and generate the eight required comparison figures.

Numbers shown in the figures come directly from CSVs saved by the experiment
code; nothing is filled in by the plotting module.
Repeated runs of the same model are summarised by their mean; error bars show
sample standard deviation across seeds.
Each figure is exported as both PNG and PDF for report insertion and zoomed layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MODEL_ORDER = ["UserCF", "ItemCF", "MF", "NeuralCF", "ContentAware"]
MODEL_LABELS = {
    "UserCF": "UserCF",
    "ItemCF": "ItemCF",
    "MF": "MF",
    "NeuralCF": "NeuralCF",
    "ContentAware": "Content-aware",
}
MODEL_COLORS = {
    "UserCF": "#7E8A97",
    "ItemCF": "#A2B2C5",
    "MF": "#537EAD",
    "NeuralCF": "#237C84",
    "ContentAware": "#E58D44",
}


def _read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    """Read a result CSV; check existence, required columns, and non-empty content."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment output: {path}")
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path.name} has no experiment rows")
    return frame


def _statistics(frame: pd.DataFrame, group: str, metric: str) -> pd.DataFrame:
    """Aggregate mean, sample standard deviation, and count by model, variant, or epoch.

    Missing records are not treated as zeros; `count` is the number of records
    that actually contributed to the metric.
    """
    values = frame[[group, metric]].copy()
    values[metric] = pd.to_numeric(values[metric], errors="raise")
    if np.isinf(values[metric].to_numpy(dtype=float)).any():
        raise ValueError(f"Infinite measurements in {metric}")
    # pandas std uses ddof=1 by default, i.e. sample standard deviation with denominator n-1.
    return values.groupby(group, sort=False)[metric].agg(["mean", "std", "count"])


def _error_values(std: Iterable[float]) -> np.ndarray | None:
    """Return None when no standard deviation is estimable, so single-seed debug figures skip error bars."""
    values = np.asarray(std, dtype=float)
    return values if np.isfinite(values).any() else None


def plot_all(results_dir: str | Path) -> list[Path]:
    """Generate the eight figures from a results directory and return PNG/PDF paths.

    The directory must contain runs.csv, loss_curves.csv, and ablation.csv.
    Metrics are aggregated across the recorded seeds; with a single run the
    sample standard deviation is undefined, so error bars are omitted. In the
    returned list, both formats of the same figure are adjacent.
    """
    import matplotlib

    # Render directly to file with no window; suitable for automatic figure generation after a full run.
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    root = Path(results_dir).resolve()
    runs = _read_csv(
        root / "runs.csv",
        [
            "model", "seed", "rmse", "mae", "sparse_user_rmse",
            "sparse_user_mae", "sparse_item_rmse", "sparse_item_mae",
        ],
    )
    # Each (model, seed) must be unique; duplicates would distort means and SDs.
    if runs.duplicated(["model", "seed"]).any():
        raise ValueError("runs.csv contains duplicate (model, seed) records")
    missing_models = set(MODEL_ORDER) - set(runs["model"])
    if missing_models:
        raise ValueError(f"runs.csv is missing models: {sorted(missing_models)}")
    curves = _read_csv(
        root / "loss_curves.csv",
        ["model", "seed", "epoch", "train_mse", "val_rmse"],
    )
    ablation = _read_csv(
        root / "ablation.csv", ["seed", "variant", "rmse", "mae", "ndcg@10"],
    )
    if ablation.duplicated(["variant", "seed"]).any():
        raise ValueError("ablation.csv contains duplicate (variant, seed) records")
    missing_variants = {"genre_on", "genre_off"} - set(ablation["variant"])
    if missing_variants:
        raise ValueError(f"ablation.csv is missing variants: {sorted(missing_variants)}")
    models = [m for m in MODEL_ORDER if m in set(runs["model"])]
    output_dir = root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#BAC3CC",
        "axes.labelcolor": "#263849",
        "xtick.color": "#3B4B5C",
        "ytick.color": "#3B4B5C",
        "text.color": "#263849",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    def finish(fig, name: str, note: str | None = None) -> None:
        """Reserve footnote space uniformly, export both formats, and close the figure to free memory."""
        if note:
            fig.text(0.5, 0.025, note, ha="center", va="bottom", fontsize=8, color="#657384")
        fig.tight_layout(rect=(0, 0.075 if note else 0.01, 1, 0.98))
        for suffix in ("png", "pdf"):
            path = output_dir / f"{name}.{suffix}"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            saved.append(path)
        plt.close(fig)

    def decorate(ax, ylabel: str) -> None:
        """Apply uniform axis and grid styling for cross-model comparison."""
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E7EBEF", linewidth=0.8)
        ax.set_ylim(bottom=0)

    # Error bars show between-seed variation, not confidence intervals, and do not replace a significance test.
    error_note = "Mean across seeds; error bars show sample SD when at least two runs are available."
    with plt.rc_context(style):
        # Rating prediction: bar height is the mean error per model; error bar is cross-seed sample SD.
        for metric, label in (("rmse", "RMSE"), ("mae", "MAE")):
            stats = _statistics(runs, "model", metric).reindex(models)
            fig, ax = plt.subplots(figsize=(8.6, 4.9))
            bars = ax.bar(
                np.arange(len(models)), stats["mean"],
                yerr=_error_values(stats["std"]), capsize=4,
                color=[MODEL_COLORS[m] for m in models], width=0.64,
                error_kw={"elinewidth": 1.2, "ecolor": "#3B4B5C"},
            )
            for bar, value, sd in zip(bars, stats["mean"], stats["std"].fillna(0)):
                if np.isfinite(value):
                    ax.annotate(
                        f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, bar.get_height() + sd),
                        xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9,
                    )
            ax.set_xticks(np.arange(len(models)), [MODEL_LABELS[m] for m in models])
            decorate(ax, label)
            ax.margins(y=0.16)
            ax.set_title(f"Rating prediction: {label} (lower is better)", pad=14)
            finish(fig, metric, error_note)

        for metric, label in (("precision", "Precision"), ("recall", "Recall"), ("ndcg", "NDCG")):
            # Read K from result column names so the x-axis matches the actual list lengths.
            ks = sorted(
                int(col.split("@", 1)[1])
                for col in runs.columns
                if col.startswith(f"{metric}@") and col.split("@", 1)[1].isdigit()
            )
            if not ks:
                raise ValueError(f"runs.csv has no {metric}@K measurements")
            fig, ax = plt.subplots(figsize=(8.6, 5.2))
            for model in models:
                rows = runs.loc[runs["model"] == model]
                means = [pd.to_numeric(rows[f"{metric}@{k}"]).mean() for k in ks]
                stds = [pd.to_numeric(rows[f"{metric}@{k}"]).std(ddof=1) for k in ks]
                ax.errorbar(
                    ks, means, yerr=_error_values(stds),
                    marker="o", markersize=5, linewidth=1.8, capsize=3,
                    color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                )
            ax.set_xticks(ks)
            ax.set_xlabel("Recommendation list length K")
            decorate(ax, f"{label}@K")
            ax.set_title(f"Top-K recommendation: {label} (higher is better)", pad=14)
            ax.legend(loc="best", frameon=False, ncol=2, fontsize=9)
            finish(fig, f"{metric}_at_k", error_note)

        # Training curves are aggregated by epoch; after early stopping some seeds
        # have no further records, so per-epoch counts may shrink.
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
        for model in ("NeuralCF", "ContentAware"):
            frame = curves.loc[curves["model"] == model]
            if frame.empty:
                raise ValueError(f"loss_curves.csv has no training records for {model}")
            for ax, metric in zip(axes, ("train_mse", "val_rmse")):
                stats = _statistics(frame, "epoch", metric).sort_index()
                x = stats.index.to_numpy(dtype=int)
                y = stats["mean"].to_numpy(dtype=float)
                sd = stats["std"].fillna(0).to_numpy(dtype=float)
                ax.plot(x, y, linewidth=1.8, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
                # The band is +/- one sample SD around the mean over available runs at that epoch.
                ax.fill_between(x, np.maximum(0, y - sd), y + sd, color=MODEL_COLORS[model], alpha=0.13)
        for ax, label, title in zip(
            axes, ("Training MSE", "Validation RMSE"), ("Training loss", "Validation performance"),
        ):
            decorate(ax, label)
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(title, pad=12)
            ax.legend(frameon=False, fontsize=9)
        finish(
            fig, "training_loss",
            "Mean +/- sample SD of available seeds at each epoch; early stopping can reduce later-epoch counts.",
        )

        # genre_off in the ablation table comes from independent retraining;
        # the same aggregation rules apply to both variants.
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.7))
        variants = ["genre_off", "genre_on"]
        for ax, metric, label, direction in zip(
            axes, ("rmse", "mae", "ndcg@10"), ("RMSE", "MAE", "NDCG@10"),
            ("lower is better", "lower is better", "higher is better"),
        ):
            stats = _statistics(ablation, "variant", metric).reindex(variants)
            bars = ax.bar(
                np.arange(2), stats["mean"], yerr=_error_values(stats["std"]),
                capsize=4, color=[MODEL_COLORS["NeuralCF"], MODEL_COLORS["ContentAware"]], width=0.6,
            )
            ax.set_xticks([0, 1], ["Genre off", "Genre on"])
            decorate(ax, label)
            ax.set_title(f"{label}\n({direction})", pad=12)
            ax.margins(y=0.19)
            for bar, value, sd in zip(bars, stats["mean"], stats["std"].fillna(0)):
                if np.isfinite(value):
                    ax.annotate(
                        f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, bar.get_height() + sd),
                        xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9,
                    )
        note = error_note
        if "max_abs_difference" in ablation:
            # Footnote confirms that independent genre-off training reproduces NeuralCF per seed.
            off_diffs = pd.to_numeric(
                ablation.loc[ablation["variant"] == "genre_off", "max_abs_difference"], errors="coerce",
            ).dropna()
            if not off_diffs.empty:
                note += (
                    "\nIndependent genre-off retraining vs NeuralCF: "
                    f"max |prediction difference| = {off_diffs.max():.3g}."
                )
        finish(fig, "genre_ablation", note)

        # Sparse groups are defined by training interaction counts; the figure only
        # shows held-out rating error by group without regrouping.
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
        positions = np.arange(len(models))
        width = 0.35
        for ax, metric, label in zip(axes, ("rmse", "mae"), ("RMSE", "MAE")):
            for offset, group, group_label, color in (
                (-width / 2, "sparse_user", "Sparse-history users", "#237C84"),
                (width / 2, "sparse_item", "Sparse-history movies", "#E58D44"),
            ):
                stats = _statistics(runs, "model", f"{group}_{metric}").reindex(models)
                ax.bar(
                    positions + offset, stats["mean"], width=width,
                    yerr=_error_values(stats["std"]), capsize=3,
                    label=group_label, color=color,
                )
            ax.set_xticks(positions, [MODEL_LABELS[m] for m in models], rotation=15, ha="right")
            decorate(ax, label)
            ax.set_title(f"Sparse-history groups: {label}", pad=12)
            ax.legend(frameon=False, fontsize=8)
        finish(
            fig, "sparse_groups",
            "Groups use training interaction counts only. Held-out errors: mean +/- sample SD across seeds.",
        )

    return saved