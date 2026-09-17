#!/usr/bin/env python3
"""Create publication figures from the current clean fullerene-gap results.

This script replaces ``draw_all_results.py``, ``draw_all_results_updated.py``,
and ``draw_test.py``.  It is intentionally independent of the training code.

Default behaviour
-----------------
* discovers one clean Proposed-model random run and one clean strict-group run;
* validates the exact split counts before accepting an automatically discovered
  run (5628/703/704 for random and 2752/3017/1266 for strict group);
* uses ``pred_raw`` as the primary paper prediction whenever it is available;
* reads the ``unit`` column and never converts an eV result a second time;
* creates per-run diagnostics and a random-versus-group comparison;
* plots the available traditional baselines, graph benchmarks, training
  histories, and ablations without pretending that incomplete runs finished;
* writes numerical plotting tables and a JSON input/output manifest.

Examples
--------
Run with automatic discovery::

    python draw_all_results_merged.py

Draw the secondary affine-calibrated sensitivity analysis::

    python draw_all_results_merged.py --prediction calibrated

Supply explicit result files::

    python draw_all_results_merged.py \
        --random-results path/to/mode_random/results.csv \
        --group-results path/to/mode_group_folder/results.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


HARTREE_TO_EV = 27.211386245988
EXPECTED_SPLIT_COUNTS = {
    "random": {"train": 5628, "val": 703, "test": 704},
    "group": {"train": 2752, "val": 3017, "test": 1266},
}
DEFAULT_SEEDS = (2024, 2025, 2026, 2027, 2028)
DEFAULT_GRAPH_MODELS = ("proposed", "fullerenenet", "matformer", "alignn")

PROTOCOL_LABELS = {
    "random": "Random split",
    "group": "Group split",
}
MODEL_LABELS = {
    "proposed": "Proposed",
    "physics_guided_residual_gnn": "Proposed",
    "dummy_train_mean": "Dummy mean",
    "ridge_size": "Ridge (size)",
    "ridge_structural": "Ridge (structural)",
    "svr_rbf_structural": "SVR-RBF",
    "random_forest_structural": "Random forest",
    "extra_trees_structural": "Extra trees",
    "hist_gradient_boosting_structural": "HistGradientBoosting",
    "fullerenenet": "FullereneNet",
    "matformer": "Matformer",
    "alignn": "ALIGNN",
}

WARNINGS: list[str] = []


def warn(message: str) -> None:
    message = str(message)
    WARNINGS.append(message)
    print(f"WARNING: {message}")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.linewidth": 1.2,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


@dataclass
class RunData:
    protocol: str
    seed: int | None
    source: Path
    frame: pd.DataFrame
    prediction_column: str
    source_unit: str
    split_counts: dict[str, int]

    @property
    def label(self) -> str:
        seed_text = f", seed {self.seed}" if self.seed is not None else ""
        return f"{PROTOCOL_LABELS[self.protocol]}{seed_text}"

    @property
    def test(self) -> pd.DataFrame:
        subset = self.frame[self.frame["split"].eq("test")].copy()
        return subset if not subset.empty else self.frame.copy()


class FigureWriter:
    def __init__(self, output_root: Path, formats: Sequence[str], dpi: int):
        self.output_root = output_root
        self.formats = tuple(dict.fromkeys(formats))
        self.dpi = int(dpi)
        self.generated: list[str] = []

    def save(self, fig: plt.Figure, relative_stem: Path | str) -> None:
        stem = self.output_root / Path(relative_stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in self.formats:
            path = stem.with_suffix(f".{suffix}")
            kwargs = {"bbox_inches": "tight"}
            if suffix.lower() == "png":
                kwargs["dpi"] = self.dpi
            fig.savefig(path, **kwargs)
            self.generated.append(str(path.resolve()))
            print(f"Saved: {path}")
        plt.close(fig)

    def write_csv(self, frame: pd.DataFrame, relative_path: Path | str) -> Path:
        path = self.output_root / Path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        self.generated.append(str(path.resolve()))
        print(f"Saved: {path}")
        return path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Draw all currently available clean fullerene HOMO-LUMO-gap results."
    )
    parser.add_argument("--project-dir", type=Path, default=script_dir)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-results", type=Path, default=None)
    parser.add_argument("--group-results", type=Path, default=None)
    parser.add_argument(
        "--prediction",
        choices=("raw", "calibrated"),
        default="raw",
        help="Raw is the primary paper result; calibrated is a secondary sensitivity analysis.",
    )
    parser.add_argument(
        "--input-unit",
        choices=("auto", "eV", "Hartree"),
        default="auto",
        help="Normally leave as auto so the unit column controls conversion.",
    )
    parser.add_argument("--gap-bins", type=int, default=8)
    parser.add_argument("--min-bin-count", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf", "svg"), default=("png", "pdf")
    )
    parser.add_argument(
        "--allow-unexpected-splits",
        action="store_true",
        help="Allow an explicitly supplied file whose split counts do not match the clean protocol.",
    )
    return parser.parse_args()


def normalise_protocol(value: object) -> str:
    text = str(value).strip().lower()
    if "random" in text:
        return "random"
    if "group" in text or "c20_56" in text:
        return "group"
    return text


def normalise_model(value: object) -> str:
    text = str(value).strip().lower()
    if text == "physics_guided_residual_gnn":
        return "proposed"
    return text


def model_label(value: object) -> str:
    key = normalise_model(value)
    return MODEL_LABELS.get(key, str(value).replace("_", " ").title())


def seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed[_-](\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def split_counts_from_frame(frame: pd.DataFrame) -> dict[str, int]:
    if "split" not in frame.columns:
        return {}
    split = frame["split"].astype(str).str.strip().str.lower()
    return {str(k): int(v) for k, v in split.value_counts().to_dict().items()}


def split_counts_from_file(path: Path) -> dict[str, int]:
    try:
        frame = pd.read_csv(path, usecols=["split"])
    except Exception:
        return {}
    return split_counts_from_frame(frame)


def counts_match(protocol: str, counts: dict[str, int]) -> bool:
    expected = EXPECTED_SPLIT_COUNTS[protocol]
    return all(int(counts.get(key, -1)) == value for key, value in expected.items())


def resolve_explicit_result(path: Path | None, project_dir: Path) -> Path | None:
    if path is None:
        return None
    path = path.expanduser()
    if not path.is_absolute():
        path = (project_dir / path).resolve()
    if path.is_dir():
        path = path / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Explicit result file was not found: {path}")
    return path


def result_candidates(project_dir: Path, protocol: str) -> list[Path]:
    mode = "mode_random" if protocol == "random" else "mode_group_folder"
    patterns = [
        f"results/graph_benchmarks/runs/proposed/{protocol}/seed_*/{mode}/results.csv",
        f"results/{protocol}/seed_*/{mode}/results.csv",
        f"results/ablations/runs/full/{protocol}/seed_*/{mode}/results.csv",
        f"raw_dataset_multiseed_benchmark/runs/proposed/{protocol}/seed_*/{mode}/results.csv",
        f"clean_module_input_ablation/runs/full/{protocol}/seed_*/{mode}/results.csv",
        f"multiseed_graph_benchmark/runs/proposed/{protocol}/seed_*/{mode}/results.csv",
        f"clean_random_and_baselines/proposed_model/{mode}/results.csv",
        f"checkpoints_final_gjf*/{mode}/results.csv",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(project_dir.glob(pattern))
    # Include other clean standalone layouts, but exclude unrelated orbital targets.
    for path in project_dir.glob(f"**/{mode}/results.csv"):
        lower = str(path).lower()
        if any(token in lower for token in ("homo_guard", "lumo_guard", "orbital_multitask")):
            continue
        paths.append(path)
    return sorted(set(path.resolve() for path in paths if path.is_file()))


def candidate_priority(path: Path) -> int:
    lower = str(path).lower().replace("\\", "/")
    if "/results/graph_benchmarks/runs/proposed/" in lower:
        score = 600
    elif "/results/random/" in lower or "/results/group/" in lower:
        score = 550
    elif "/raw_dataset_multiseed_benchmark/" in lower:
        score = 500
    elif "/clean_module_input_ablation/" in lower and "/runs/full/" in lower:
        score = 450
    elif "checkpoints_final_gjf" in lower:
        score = 400
    elif "/clean_random_and_baselines/proposed_model/" in lower:
        score = 300
    elif "/multiseed_graph_benchmark/" in lower:
        score = 200
    else:
        score = 100
    if " (2)" in str(path):
        score += 20
    return score


def discover_result(project_dir: Path, protocol: str) -> Path | None:
    valid: list[tuple[int, float, Path]] = []
    rejected: list[tuple[Path, dict[str, int]]] = []
    for path in result_candidates(project_dir, protocol):
        counts = split_counts_from_file(path)
        if counts_match(protocol, counts):
            valid.append((candidate_priority(path), path.stat().st_mtime, path))
        else:
            rejected.append((path, counts))
    if rejected:
        for path, counts in rejected:
            warn(f"Rejected non-clean {protocol} result {path}: split counts {counts}.")
    if not valid:
        warn(f"No clean {protocol} Proposed-model results.csv was found under {project_dir}.")
        return None
    valid.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = valid[0][2]
    print(f"Selected {protocol} result: {selected}")
    return selected


def choose_prediction_column(frame: pd.DataFrame, variant: str, source: Path) -> str:
    if variant == "raw":
        for name in ("pred_raw", "y_pred_raw"):
            if name in frame.columns:
                return name
        if "pred" in frame.columns:
            warn(f"{source} has no pred_raw column; falling back to pred.")
            return "pred"
    else:
        for name in ("pred_calibrated", "y_pred_calibrated"):
            if name in frame.columns:
                return name
        if "pred" in frame.columns:
            warn(f"{source} has no pred_calibrated column; falling back to pred.")
            return "pred"
    raise ValueError(f"No usable {variant} prediction column was found in {source}.")


def determine_scale(frame: pd.DataFrame, input_unit: str, source: Path) -> tuple[float, str]:
    if input_unit != "auto":
        return (1.0, "eV") if input_unit == "eV" else (HARTREE_TO_EV, "Hartree")
    if "unit" in frame.columns:
        values = frame["unit"].dropna().astype(str).str.strip().str.lower().unique().tolist()
        if len(values) == 1:
            unit = values[0]
            if unit in {"ev", "electronvolt", "electronvolts"}:
                return 1.0, "eV"
            if unit in {"hartree", "ha", "a.u.", "au"}:
                return HARTREE_TO_EV, "Hartree"
        elif len(values) > 1:
            raise ValueError(f"Mixed units in {source}: {values}")
    true_col = "true" if "true" in frame.columns else "y_true"
    typical = float(frame[true_col].abs().quantile(0.95))
    if typical < 0.25:
        warn(f"{source} has no reliable unit field; values look like Hartree and will be converted.")
        return HARTREE_TO_EV, "inferred Hartree"
    warn(f"{source} has no reliable unit field; values are assumed to be eV.")
    return 1.0, "inferred eV"


def load_run(
    path: Path,
    protocol: str,
    prediction: str,
    input_unit: str,
    allow_unexpected_splits: bool,
) -> RunData:
    raw = pd.read_csv(path)
    true_col = "true" if "true" in raw.columns else "y_true" if "y_true" in raw.columns else None
    n_col = "N" if "N" in raw.columns else "atom_size" if "atom_size" in raw.columns else None
    if true_col is None or n_col is None or "split" not in raw.columns:
        raise ValueError(f"{path} must contain true/y_true, N/atom_size, and split columns.")
    prediction_col = choose_prediction_column(raw, prediction, path)
    scale, source_unit = determine_scale(raw, input_unit, path)
    frame = pd.DataFrame(
        {
            "y_true": pd.to_numeric(raw[true_col], errors="coerce") * scale,
            "y_pred": pd.to_numeric(raw[prediction_col], errors="coerce") * scale,
            "N": pd.to_numeric(raw[n_col], errors="coerce"),
            "split": raw["split"].astype(str).str.strip().str.lower(),
        }
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred", "N"])
    frame["N"] = frame["N"].astype(int)
    frame["residual"] = frame["y_pred"] - frame["y_true"]
    frame["abs_error"] = frame["residual"].abs()
    counts = split_counts_from_frame(frame)
    if not counts_match(protocol, counts):
        message = (
            f"{path} has split counts {counts}; expected {EXPECTED_SPLIT_COUNTS[protocol]} "
            f"for the clean {protocol} protocol."
        )
        if not allow_unexpected_splits:
            raise ValueError(message)
        warn(message)
    return RunData(
        protocol=protocol,
        seed=seed_from_path(path),
        source=path,
        frame=frame,
        prediction_column=prediction_col,
        source_unit=source_unit,
        split_counts=counts,
    )


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred"])
    if frame.empty:
        return {"r2": math.nan, "mae": math.nan, "rmse": math.nan, "bias": math.nan}
    true = frame["y_true"].to_numpy(dtype=float)
    pred = frame["y_pred"].to_numpy(dtype=float)
    error = pred - true
    denominator = float(np.square(true - true.mean()).sum())
    r2 = math.nan if len(true) < 2 or denominator <= 0 else 1.0 - float(np.square(error).sum()) / denominator
    return {
        "r2": float(r2),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
    }


def metric_row(run: RunData, split: str = "test") -> dict[str, object]:
    subset = run.frame[run.frame["split"].eq(split)]
    stats = metric_values(subset)
    return {
        "source_run": str(run.source.resolve()),
        "model": "proposed",
        "protocol": run.protocol,
        "seed": run.seed,
        "split": split,
        "prediction_variant": "raw" if "raw" in run.prediction_column else "calibrated",
        "count": int(len(subset)),
        "unit": "eV",
        **stats,
    }


def metrics_by_n(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, n_value), subset in frame.groupby(["split", "N"], sort=True):
        rows.append({"split": split, "N": int(n_value), "count": len(subset), **metric_values(subset)})
    return pd.DataFrame(rows)


def limits_for(frame: pd.DataFrame) -> tuple[float, float]:
    values = np.concatenate([frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()])
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = 0.08 * (hi - lo) if hi > lo else 0.1
    return lo - pad, hi + pad


def set_smart_ticks(ax: plt.Axes, lo: float, hi: float) -> None:
    span = hi - lo
    if span > 4:
        step = 1.0
    elif span > 2:
        step = 0.5
    elif span > 1:
        step = 0.25
    else:
        step = 0.2
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))


def scatter_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    show_ylabel: bool = True,
    norm: mcolors.Normalize | None = None,
):
    lo, hi = limits_for(frame)
    if norm is None:
        n_min, n_max = int(frame["N"].min()), int(frame["N"].max())
        norm = mcolors.Normalize(vmin=n_min, vmax=n_max)
    scatter = ax.scatter(
        frame["y_true"],
        frame["y_pred"],
        c=frame["N"],
        cmap="plasma",
        norm=norm,
        s=24,
        alpha=0.82,
        edgecolors="black",
        linewidths=0.15,
        rasterized=True,
    )
    ax.plot([lo, hi], [lo, hi], "--", color="#333333", linewidth=1.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    set_smart_ticks(ax, lo, hi)
    ax.set_xlabel("DFT-calculated HOMO-LUMO gap (eV)")
    if show_ylabel:
        ax.set_ylabel("Predicted HOMO-LUMO gap (eV)")
    ax.set_title(title, fontweight="bold")
    stats = metric_values(frame)
    text = (
        f"$R^2$ = {stats['r2']:.4f}\n"
        f"MAE = {stats['mae']:.4f} eV\n"
        f"RMSE = {stats['rmse']:.4f} eV\n"
        f"$n$ = {len(frame)}"
    )
    ax.text(0.96, 0.05, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    return scatter


def plot_single_scatter(run: RunData, writer: FigureWriter, run_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    scatter = scatter_panel(ax, run.test, run.label)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Number of carbon atoms, $N$")
    fig.tight_layout()
    writer.save(fig, run_root / "prediction_scatter_test")


def plot_combined_scatter(runs: dict[str, RunData], writer: FigureWriter, prediction: str) -> None:
    available = [runs[key] for key in ("random", "group") if key in runs]
    if not available:
        return
    n_min = min(int(run.test["N"].min()) for run in available)
    n_max = max(int(run.test["N"].max()) for run in available)
    shared_norm = mcolors.Normalize(vmin=n_min, vmax=n_max)
    fig = plt.figure(figsize=(7.0 * len(available) + 0.7, 5.8), layout="constrained")
    grid = fig.add_gridspec(
        1,
        len(available) + 1,
        width_ratios=[1.0] * len(available) + [0.045],
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(len(available))]
    colorbar_axis = fig.add_subplot(grid[0, -1])
    scatters = []
    for index, run in enumerate(available):
        scatters.append(
            scatter_panel(
                axes[index],
                run.test,
                run.label,
                show_ylabel=index == 0,
                norm=shared_norm,
            )
        )
    cbar = fig.colorbar(scatters[-1], cax=colorbar_axis)
    cbar.set_label("Number of carbon atoms, $N$")
    fig.suptitle(f"Proposed model: {prediction} predictions", fontweight="bold")
    writer.save(fig, Path("comparisons") / f"proposed_random_vs_group_{prediction}")


def plot_residuals(run: RunData, writer: FigureWriter, run_root: Path) -> None:
    frame = run.test
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    norm = mcolors.Normalize(vmin=int(frame["N"].min()), vmax=int(frame["N"].max()))
    scatter = ax.scatter(
        frame["y_true"], frame["residual"], c=frame["N"], cmap="plasma", norm=norm,
        s=23, alpha=0.82, edgecolors="black", linewidths=0.15, rasterized=True
    )
    ax.axhline(0, color="#333333", linestyle="--", linewidth=1.5)
    ax.set_xlabel("DFT-calculated HOMO-LUMO gap (eV)")
    ax.set_ylabel("Residual: predicted - DFT (eV)")
    ax.set_title(f"Residual distribution - {run.label}", fontweight="bold")
    ax.grid(alpha=0.16)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Number of carbon atoms, $N$")
    fig.tight_layout()
    writer.save(fig, run_root / "residual_scatter_test")


def plot_distributions(run: RunData, writer: FigureWriter, run_root: Path) -> None:
    frame = run.test
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    bins = np.histogram_bin_edges(np.concatenate([frame["y_true"], frame["y_pred"]]), bins=32)
    axes[0].hist(frame["y_true"], bins=bins, density=True, alpha=0.58, label="DFT", color="#3B6FB6")
    axes[0].hist(frame["y_pred"], bins=bins, density=True, alpha=0.55, label="Prediction", color="#D55E00")
    axes[0].set_xlabel("HOMO-LUMO gap (eV)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Target and prediction distributions", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[1].hist(frame["residual"], bins=32, alpha=0.72, color="#6A4C93", label="Residual")
    axes[1].hist(frame["abs_error"], bins=32, alpha=0.55, color="#2A9D8F", label="Absolute error")
    axes[1].axvline(0, color="#333333", linestyle="--", linewidth=1.3)
    axes[1].set_xlabel("Error (eV)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Error distributions", fontweight="bold")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(run.label, fontweight="bold")
    fig.tight_layout()
    writer.save(fig, run_root / "value_and_error_distributions_test")


def plot_density(run: RunData, writer: FigureWriter, run_root: Path) -> None:
    frame = run.test
    lo, hi = limits_for(frame)
    fig, ax = plt.subplots(figsize=(6.7, 5.7))
    hist = ax.hist2d(
        frame["y_true"], frame["y_pred"], bins=30, range=[[lo, hi], [lo, hi]],
        cmap="viridis", norm=mcolors.LogNorm(vmin=1)
    )
    ax.plot([lo, hi], [lo, hi], "--", color="white", linewidth=1.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("DFT-calculated HOMO-LUMO gap (eV)")
    ax.set_ylabel("Predicted HOMO-LUMO gap (eV)")
    ax.set_title(f"Prediction density - {run.label}", fontweight="bold")
    cbar = fig.colorbar(hist[3], ax=ax)
    cbar.set_label("Sample count")
    fig.tight_layout()
    writer.save(fig, run_root / "prediction_density_test")


def rank_profile(frame: pd.DataFrame, value_column: str, bins: int = 50) -> tuple[np.ndarray, list[int]]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    target_x = np.linspace(0.0, 1.0, bins)
    for n_value, group in frame.groupby("N", sort=True):
        ordered = group.sort_values("y_true")
        values = ordered[value_column].to_numpy(dtype=float)
        if values.size == 0:
            continue
        if values.size == 1:
            interpolated = np.repeat(values[0], bins)
        else:
            interpolated = np.interp(target_x, np.linspace(0.0, 1.0, values.size), values)
        rows.append(interpolated)
        labels.append(int(n_value))
    return (np.vstack(rows), labels) if rows else (np.empty((0, bins)), [] )


def plot_rank_profiles(run: RunData, writer: FigureWriter, run_root: Path) -> None:
    frame = run.test
    true_matrix, labels = rank_profile(frame, "y_true")
    pred_matrix, _ = rank_profile(frame, "y_pred")
    error_matrix, _ = rank_profile(frame, "abs_error")
    if true_matrix.size == 0:
        return
    gap_lo = float(min(true_matrix.min(), pred_matrix.min()))
    gap_hi = float(max(true_matrix.max(), pred_matrix.max()))
    error_hi = float(np.nanpercentile(error_matrix, 98))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, max(5.0, 0.23 * len(labels))))
    panels = [
        (true_matrix, "DFT-calculated", "YlOrRd", gap_lo, gap_hi, "Gap (eV)"),
        (pred_matrix, "Predicted", "YlOrRd", gap_lo, gap_hi, "Gap (eV)"),
        (error_matrix, "Absolute error", "magma", 0.0, error_hi, "Absolute error (eV)"),
    ]
    for ax, (matrix, title, cmap, lo, hi, cbar_label) in zip(axes, panels):
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Isomer percentile within each cage size")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_ylabel("Number of carbon atoms, $N$")
        ax.set_xticks([0, 12, 24, 36, 49])
        ax.set_xticklabels(["0", "25", "50", "75", "100"])
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(cbar_label)
    fig.suptitle(run.label, fontweight="bold")
    fig.tight_layout()
    writer.save(fig, run_root / "rank_profile_heatmaps_test")


def plot_error_by_n(run: RunData, writer: FigureWriter, run_root: Path) -> pd.DataFrame:
    table = metrics_by_n(run.test.assign(split="test"))
    if table.empty:
        return table
    x = np.arange(len(table))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8.5, 0.55 * len(table)), 5.0))
    ax.bar(x - width / 2, table["mae"], width, label="MAE", color="#3B6FB6")
    ax.bar(x + width / 2, table["rmse"], width, label="RMSE", color="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels(table["N"].astype(int), rotation=45, ha="right")
    ax.set_xlabel("Number of carbon atoms, $N$")
    ax.set_ylabel("Error (eV)")
    ax.set_title(f"Test error by cage size - {run.label}", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.16)
    fig.tight_layout()
    writer.save(fig, run_root / "mae_rmse_by_cage_size_test")
    writer.write_csv(table, run_root / "metrics_by_cage_size_test.csv")
    return table


def size_ranges(protocol: str) -> tuple[tuple[int, int, str], ...]:
    if protocol == "group":
        return ((70, 80, "C70-C80"), (82, 90, "C82-C90"), (92, 100, "C92-C100"))
    return (
        (20, 30, "C20-C30"), (32, 40, "C32-C40"), (42, 50, "C42-C50"),
        (52, 60, "C52-C60"), (70, 80, "C70-C80"), (82, 90, "C82-C90"),
        (92, 100, "C92-C100"),
    )


def assign_size_group(n_value: int, protocol: str) -> str:
    for low, high, label in size_ranges(protocol):
        if low <= int(n_value) <= high:
            return label
    return f"C{int(n_value)}"


def annotate_matrix(ax: plt.Axes, matrix: np.ndarray, count_mode: bool, vmax: float) -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isfinite(value):
                continue
            label = f"{int(value)}" if count_mode else f"{value:.3f}"
            threshold = 0.55 * vmax if vmax > 0 else 0.0
            color = "white" if value < threshold else "black"
            ax.text(col, row, label, ha="center", va="center", fontsize=7, color=color)


def plot_gap_bin_heatmaps(
    run: RunData,
    writer: FigureWriter,
    run_root: Path,
    gap_bins: int,
    min_count: int,
) -> None:
    frame = run.test.copy()
    lo, hi = float(frame["y_true"].min()), float(frame["y_true"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return
    edges = np.linspace(lo, hi, gap_bins + 1)
    labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(gap_bins)]
    frame["gap_bin"] = pd.cut(frame["y_true"], bins=edges, labels=labels, include_lowest=True)
    frame["size_group"] = frame["N"].map(lambda value: assign_size_group(int(value), run.protocol))
    order = [label for _, _, label in size_ranges(run.protocol)]
    observed_groups = frame["size_group"].dropna().unique().tolist()
    order = [label for label in order if label in observed_groups] + [
        label for label in observed_groups if label not in order
    ]
    count = frame.pivot_table(
        index="size_group", columns="gap_bin", values="abs_error", aggfunc="size",
        observed=False, fill_value=0
    ).reindex(index=order, columns=labels).fillna(0).astype(int)
    mae = frame.pivot_table(
        index="size_group", columns="gap_bin", values="abs_error", aggfunc="mean", observed=False
    ).reindex(index=order, columns=labels)
    stable_mae = mae.where(count >= min_count)
    writer.write_csv(stable_mae.reset_index(), run_root / "mae_by_size_and_gap_bin.csv")
    writer.write_csv(count.reset_index(), run_root / "sample_count_by_size_and_gap_bin.csv")

    mae_matrix = stable_mae.to_numpy(dtype=float)
    count_matrix = count.to_numpy(dtype=float)
    finite_mae = mae_matrix[np.isfinite(mae_matrix)]
    mae_vmax = float(np.nanpercentile(finite_mae, 95)) if finite_mae.size else 1.0
    count_vmax = float(np.nanmax(count_matrix)) if count_matrix.size else 1.0
    mae_cmap = plt.get_cmap("magma").copy()
    mae_cmap.set_bad("white")
    count_cmap = plt.get_cmap("viridis").copy()
    count_display = np.where(count_matrix > 0, count_matrix, np.nan)
    count_cmap.set_bad("white")

    fig, axes = plt.subplots(1, 2, figsize=(15.0, max(4.8, 0.55 * len(order))))
    image0 = axes[0].imshow(
        np.ma.masked_invalid(mae_matrix), aspect="auto", cmap=mae_cmap, vmin=0, vmax=mae_vmax
    )
    image1 = axes[1].imshow(
        np.ma.masked_invalid(count_display), aspect="auto", cmap=count_cmap, vmin=1, vmax=max(1, count_vmax)
    )
    for ax, title in zip(axes, ("Mean absolute error", "Test-sample count")):
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlabel("DFT-calculated gap range (eV)")
        ax.set_ylabel("Cage-size range")
        ax.set_title(title, fontweight="bold")
    annotate_matrix(axes[0], mae_matrix, False, mae_vmax)
    annotate_matrix(axes[1], count_display, True, count_vmax)
    cbar0 = fig.colorbar(image0, ax=axes[0])
    cbar0.set_label("MAE (eV)")
    cbar1 = fig.colorbar(image1, ax=axes[1])
    cbar1.set_label("Sample count")
    fig.suptitle(
        f"{run.label}; blank MAE cells contain fewer than {min_count} samples",
        fontweight="bold",
    )
    fig.tight_layout()
    writer.save(fig, run_root / "mae_and_count_by_size_gap_bin_test")


def plot_tail_metrics(run: RunData, writer: FigureWriter, run_root: Path) -> None:
    frame = run.test
    threshold = float(frame["y_true"].quantile(0.90))
    subsets = {"Overall": frame, "High-gap top 10%": frame[frame["y_true"] >= threshold]}
    rows = []
    for label, subset in subsets.items():
        rows.append({"segment": label, "count": len(subset), "threshold_eV": threshold, **metric_values(subset)})
    table = pd.DataFrame(rows)
    writer.write_csv(table, run_root / "overall_and_high_gap_metrics.csv")
    x = np.arange(len(table))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    for ax, column, ylabel, color in (
        (axes[0], "mae", "MAE (eV)", "#3B6FB6"),
        (axes[1], "rmse", "RMSE (eV)", "#D55E00"),
        (axes[2], "bias", "Bias (eV)", "#2A9D8F"),
    ):
        ax.bar(x, table[column], color=color, alpha=0.86)
        if column == "bias":
            ax.axhline(0, color="#333333", linestyle="--", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(table["segment"], rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(f"Overall and high-gap errors - {run.label}", fontweight="bold")
    fig.tight_layout()
    writer.save(fig, run_root / "overall_vs_high_gap_errors_test")


def plot_split_counts(run: RunData, writer: FigureWriter, run_root: Path) -> None:
    table = run.frame.groupby(["N", "split"], as_index=False).size().rename(columns={"size": "count"})
    pivot = table.pivot(index="N", columns="split", values="count").fillna(0)
    order = [column for column in ("train", "val", "test") if column in pivot.columns]
    pivot = pivot[order]
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_xlabel("Number of carbon atoms, $N$")
    ax.set_ylabel("Sample count")
    ax.set_title(f"Dataset split composition - {run.label}", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    writer.save(fig, run_root / "split_counts_by_cage_size")
    writer.write_csv(table, run_root / "split_counts_by_cage_size.csv")


def last_run_status_block(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("START split=")]
    return lines[starts[-1]:] if starts else lines


def parse_status_history(path: Path, expected_counts: dict[str, int]) -> pd.DataFrame:
    lines = last_run_status_block(path)
    if not lines:
        return pd.DataFrame()
    split_line = next((line for line in lines if line.startswith("SPLIT ")), "")
    match = re.search(r"train=(\d+)\s+val=(\d+)\s+test=(\d+)", split_line)
    if match:
        status_counts = dict(zip(("train", "val", "test"), map(int, match.groups())))
        if status_counts != expected_counts:
            warn(f"Skipped mismatched training log {path}: {status_counts} != {expected_counts}.")
            return pd.DataFrame()
    records: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    numeric = re.compile(r"^([A-Za-z0-9_]+)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$")
    for line in lines:
        if line.startswith("EPOCH="):
            if current:
                records.append(current)
            current = {"epoch": float(line.split("=", 1)[1])}
            continue
        found = numeric.match(line)
        if found and current is not None and found.group(1) in {
            "loss", "val_score", "val_r2_overall", "best_score"
        }:
            current[found.group(1)] = float(found.group(2))
    if current:
        records.append(current)
    return pd.DataFrame(records)


def plot_training_status(run: RunData, writer: FigureWriter, run_root: Path) -> None:
    status_path = run.source.parent / "run_status.txt"
    history = parse_status_history(status_path, run.split_counts)
    if history.empty:
        return
    writer.write_csv(history, run_root / "training_history_from_run_status.csv")
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    lines = []
    if "loss" in history.columns:
        lines += ax1.plot(history["epoch"], history["loss"], color="#3B6FB6", label="Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="#3B6FB6")
    ax1.tick_params(axis="y", labelcolor="#3B6FB6")
    ax2 = ax1.twinx()
    if "val_r2_overall" in history.columns:
        lines += ax2.plot(
            history["epoch"], history["val_r2_overall"], color="#D55E00", label="Validation $R^2$"
        )
    if "val_score" in history.columns:
        lines += ax2.plot(
            history["epoch"], history["val_score"], color="#2A9D8F", linestyle="--",
            label="Validation selection score"
        )
    ax2.set_ylabel("Validation metric", color="#D55E00")
    ax2.tick_params(axis="y", labelcolor="#D55E00")
    if lines:
        ax1.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")
    ax1.set_title(f"Training history - {run.label}", fontweight="bold")
    ax1.grid(alpha=0.14)
    fig.tight_layout()
    writer.save(fig, run_root / "training_history")


def run_output_root(run: RunData, prediction: str) -> Path:
    seed_text = f"seed_{run.seed}" if run.seed is not None else "selected_run"
    return Path("proposed") / run.protocol / seed_text / prediction


def draw_run(
    run: RunData,
    writer: FigureWriter,
    prediction: str,
    gap_bins: int,
    min_count: int,
) -> None:
    root = run_output_root(run, prediction)
    writer.write_csv(pd.DataFrame([metric_row(run)]), root / "test_metrics.csv")
    plot_single_scatter(run, writer, root)
    plot_residuals(run, writer, root)
    plot_distributions(run, writer, root)
    plot_density(run, writer, root)
    plot_rank_profiles(run, writer, root)
    plot_error_by_n(run, writer, root)
    plot_gap_bin_heatmaps(run, writer, root, gap_bins, min_count)
    plot_tail_metrics(run, writer, root)
    plot_split_counts(run, writer, root)
    plot_training_status(run, writer, root)


def metric_scale_from_unit(unit: object) -> float:
    text = str(unit).strip().lower()
    return HARTREE_TO_EV if text in {"hartree", "ha", "au", "a.u."} else 1.0


def normalise_metric_units(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "unit" not in frame.columns:
        return frame
    scales = frame["unit"].map(metric_scale_from_unit).astype(float)
    for column in ("mae", "rmse", "bias", "bias_pred_minus_true", "true_mean", "pred_mean"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce") * scales
    frame["unit"] = "eV"
    return frame


def load_traditional_baselines(project_dir: Path, runs: dict[str, RunData]) -> pd.DataFrame:
    path = project_dir / "clean_random_and_baselines" / "test_metrics_comparison.csv"
    frames: list[pd.DataFrame] = []
    if path.exists():
        base = normalise_metric_units(pd.read_csv(path))
        base = base[base.get("split", "test").astype(str).str.lower().eq("test")].copy()
        base["model"] = base["model"].map(normalise_model)
        base = base[base["model"] != "proposed"]
        base["protocol"] = base["protocol"].map(normalise_protocol)
        base["source_run"] = str(path.resolve())
        base["prediction_variant"] = "raw"
        frames.append(base)
    else:
        warn(f"Traditional baseline summary was not found: {path}")
    if runs:
        frames.append(pd.DataFrame([metric_row(run) for run in runs.values()]))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["model_label"] = combined["model"].map(model_label)
    for column in ("r2", "mae", "rmse"):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


def plot_model_comparison(
    frame: pd.DataFrame,
    writer: FigureWriter,
    relative_stem: Path,
    title_prefix: str,
    use_errorbars: bool = False,
) -> None:
    if frame.empty:
        return
    for protocol, subset in frame.groupby("protocol", sort=False):
        subset = subset.dropna(subset=["r2", "mae", "rmse"]).sort_values("r2", ascending=True)
        if subset.empty:
            continue
        labels = subset["model_label"].astype(str).tolist()
        y = np.arange(len(subset))
        fig, axes = plt.subplots(1, 3, figsize=(15.5, max(4.8, 0.52 * len(subset))))
        specs = (("r2", "$R^2$", "#3B6FB6"), ("mae", "MAE (eV)", "#D55E00"), ("rmse", "RMSE (eV)", "#2A9D8F"))
        for index, (ax, (column, xlabel, color)) in enumerate(zip(axes, specs)):
            errors = None
            std_col = f"{column}_std"
            if use_errorbars and std_col in subset.columns:
                errors = subset[std_col].fillna(0.0).to_numpy(dtype=float)
            ax.barh(y, subset[column], xerr=errors, capsize=3, color=color, alpha=0.86)
            ax.axvline(0, color="#333333", linewidth=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels(labels if index == 0 else [])
            ax.set_xlabel(xlabel)
            ax.set_title(xlabel, fontweight="bold")
            ax.grid(axis="x", alpha=0.15)
            if "n_seeds" in subset.columns:
                for yi, (_, row) in zip(y, subset.iterrows()):
                    ax.text(row[column], yi, f"  n={int(row['n_seeds'])}", va="center", fontsize=7)
        fig.suptitle(f"{title_prefix} - {PROTOCOL_LABELS.get(protocol, protocol)}", fontweight="bold")
        fig.tight_layout()
        writer.save(fig, relative_stem / f"{protocol}_test_metrics")


def collect_graph_metrics(project_dir: Path) -> tuple[pd.DataFrame, Path | None]:
    primary = project_dir / "results" / "graph_benchmarks"
    source_root: Path | None = primary if primary.exists() else None
    if source_root is None:
        legacy = project_dir / "multiseed_graph_benchmark"
        if legacy.exists():
            source_root = legacy
            warn("Using the older multiseed_graph_benchmark because no raw-dataset benchmark directory exists.")
    if source_root is None:
        warn("No graph benchmark directory was found.")
        return pd.DataFrame(), None
    frames = []
    for path in sorted(source_root.glob("runs/*/*/seed_*/metrics.csv")):
        try:
            frame = normalise_metric_units(pd.read_csv(path))
        except Exception as exc:
            warn(f"Could not read graph metric file {path}: {exc}")
            continue
        if "split" not in frame.columns:
            continue
        frame = frame[frame["split"].astype(str).str.lower().eq("test")].copy()
        if frame.empty:
            continue
        if "model" not in frame.columns:
            frame["model"] = path.parents[2].name
        if "protocol" not in frame.columns:
            frame["protocol"] = path.parents[1].name
        if "seed" not in frame.columns:
            frame["seed"] = seed_from_path(path)
        frame["model"] = frame["model"].map(normalise_model)
        frame["protocol"] = frame["protocol"].map(normalise_protocol)
        frame["source_run"] = str(path.resolve())
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), source_root
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["model", "protocol", "seed", "split"], keep="last")
    combined["model_label"] = combined["model"].map(model_label)
    return combined, source_root


def summarise_repeated_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (model, protocol), subset in frame.groupby(["model", "protocol"]):
        row: dict[str, object] = {
            "model": model,
            "model_label": model_label(model),
            "protocol": protocol,
            "n_seeds": int(subset["seed"].nunique()),
            "seeds": ",".join(str(int(value)) for value in sorted(subset["seed"].dropna().unique())),
            "unit": "eV",
        }
        for metric in ("r2", "mae", "rmse"):
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            row[metric] = float(values.mean()) if len(values) else math.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def graph_completeness(frame: pd.DataFrame) -> pd.DataFrame:
    completed = {
        (normalise_model(row.model), normalise_protocol(row.protocol), int(row.seed))
        for row in frame.itertuples(index=False)
        if pd.notna(row.seed)
    } if not frame.empty else set()
    rows = []
    for model in DEFAULT_GRAPH_MODELS:
        for protocol in ("random", "group"):
            for seed in DEFAULT_SEEDS:
                rows.append(
                    {
                        "model": model,
                        "protocol": protocol,
                        "seed": seed,
                        "test_metrics_available": (model, protocol, seed) in completed,
                    }
                )
    return pd.DataFrame(rows)


def plot_external_training_histories(project_dir: Path, writer: FigureWriter) -> None:
    root = project_dir / "results" / "graph_benchmarks"
    if not root.exists():
        return
    for path in sorted(root.glob("runs/*/*/seed_*/training_history.csv")):
        try:
            history = pd.read_csv(path)
        except Exception as exc:
            warn(f"Could not read training history {path}: {exc}")
            continue
        if history.empty or "epoch" not in history.columns:
            continue
        model = path.parents[2].name
        protocol = normalise_protocol(path.parents[1].name)
        seed = seed_from_path(path)
        fig, ax1 = plt.subplots(figsize=(7.4, 4.8))
        lines = []
        if "train_loss" in history.columns:
            lines += ax1.plot(history["epoch"], history["train_loss"], color="#3B6FB6", label="Training loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training loss", color="#3B6FB6")
        ax1.tick_params(axis="y", labelcolor="#3B6FB6")
        ax2 = ax1.twinx()
        for column, label, style in (
            ("validation_r2", "Validation $R^2$", "-"),
            ("validation_selection_score", "Validation score", "--"),
        ):
            if column in history.columns:
                lines += ax2.plot(history["epoch"], history[column], linestyle=style, label=label)
        ax2.set_ylabel("Validation metric")
        if lines:
            ax1.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")
        ax1.set_title(
            f"{model_label(model)} training history - {PROTOCOL_LABELS.get(protocol, protocol)}, seed {seed}",
            fontweight="bold",
        )
        ax1.grid(alpha=0.14)
        fig.tight_layout()
        writer.save(fig, Path("training_histories") / model / protocol / f"seed_{seed}")


def collect_ablation_metrics(project_dir: Path) -> pd.DataFrame:
    root = project_dir / "results" / "ablations"
    summary_path = root / "test_metrics_mean_std.csv"
    if summary_path.exists():
        frame = pd.read_csv(summary_path)
        frame["protocol"] = frame["protocol"].map(normalise_protocol)
        return frame
    frames = []
    for path in sorted(root.glob("runs/*/*/seed_*/metrics.csv")) if root.exists() else []:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "split" in frame.columns:
            frame = frame[frame["split"].astype(str).str.lower().eq("test")].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        warn("No completed clean module/input ablation metrics were found.")
        return pd.DataFrame()
    per_seed = pd.concat(frames, ignore_index=True, sort=False)
    rows = []
    for (family, config, protocol), subset in per_seed.groupby(["family", "config", "protocol"]):
        row = {
            "family": family,
            "config": config,
            "protocol": normalise_protocol(protocol),
            "n_seeds": int(subset["seed"].nunique()),
        }
        for short, source in (("r2", "r2_raw"), ("mae", "mae_raw"), ("rmse", "rmse_raw")):
            values = pd.to_numeric(subset[source], errors="coerce").dropna()
            row[short] = float(values.mean()) if len(values) else math.nan
            row[f"{short}_std"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_ablations(frame: pd.DataFrame, writer: FigureWriter) -> None:
    if frame.empty:
        return
    frame = frame.copy()
    rename = {
        "r2_raw_mean": "r2", "r2_raw_std": "r2_std",
        "mae_raw_mean": "mae", "mae_raw_std": "mae_std",
        "rmse_raw_mean": "rmse", "rmse_raw_std": "rmse_std",
    }
    frame = frame.rename(columns=rename)
    for (family, protocol), subset in frame.groupby(["family", "protocol"]):
        subset = subset.dropna(subset=["r2"]).copy()
        if subset.empty:
            continue
        x = np.arange(len(subset))
        error = subset.get("r2_std", pd.Series(0.0, index=subset.index)).fillna(0.0)
        fig, ax = plt.subplots(figsize=(max(8.0, 1.25 * len(subset)), 5.0))
        ax.bar(x, subset["r2"], yerr=error, capsize=4, color="#4472C4", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(subset["config"], rotation=25, ha="right")
        ax.set_ylabel("Test $R^2$ (raw prediction)")
        ax.set_title(f"{family.title()} ablation - {PROTOCOL_LABELS.get(protocol, protocol)}", fontweight="bold")
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.grid(axis="y", alpha=0.18)
        fig.tight_layout()
        writer.save(fig, Path("ablations") / family / f"{protocol}_raw_r2_mean_std")


def write_text_report(
    writer: FigureWriter,
    runs: dict[str, RunData],
    graph_metrics: pd.DataFrame,
    completeness: pd.DataFrame,
    ablations: pd.DataFrame,
) -> None:
    lines = ["Fullerene plotting report", "=========================", ""]
    for protocol in ("random", "group"):
        if protocol in runs:
            run = runs[protocol]
            stats = metric_values(run.test)
            lines.extend(
                [
                    f"{PROTOCOL_LABELS[protocol]}:",
                    f"  source: {run.source}",
                    f"  prediction column: {run.prediction_column}",
                    f"  split counts: {run.split_counts}",
                    f"  test R2: {stats['r2']:.6f}",
                    f"  test MAE: {stats['mae']:.6f} eV",
                    f"  test RMSE: {stats['rmse']:.6f} eV",
                    "",
                ]
            )
        else:
            lines.extend([f"{PROTOCOL_LABELS[protocol]}: no clean result found", ""])
    completed = int(completeness["test_metrics_available"].sum()) if not completeness.empty else 0
    lines.append(f"Graph benchmark test-metric runs available: {completed}/{len(completeness)}")
    if not graph_metrics.empty:
        lines.append("Available graph test metrics:")
        for row in graph_metrics.itertuples(index=False):
            lines.append(f"  {row.model}/{row.protocol}/seed_{int(row.seed)}")
    lines.append("")
    lines.append(f"Ablation summary rows available: {len(ablations)}")
    lines.append("")
    lines.append("Warnings:")
    lines.extend([f"  - {message}" for message in WARNINGS] or ["  none"])
    path = writer.output_root / "plot_report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    writer.generated.append(str(path.resolve()))


def main() -> None:
    args = parse_args()
    configure_style()
    project_dir = args.project_dir.expanduser().resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory does not exist: {project_dir}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_dir / "results" / "paper_figures" / args.prediction
    )
    writer = FigureWriter(output_dir, args.formats, args.dpi)
    runs: dict[str, RunData] = {}

    explicit = {
        "random": resolve_explicit_result(args.random_results, project_dir),
        "group": resolve_explicit_result(args.group_results, project_dir),
    }
    for protocol in ("random", "group"):
        path = explicit[protocol] or discover_result(project_dir, protocol)
        if path is None:
            continue
        runs[protocol] = load_run(
            path, protocol, args.prediction, args.input_unit, args.allow_unexpected_splits
        )
        draw_run(runs[protocol], writer, args.prediction, args.gap_bins, args.min_bin_count)

    if not runs:
        raise RuntimeError("No clean Proposed-model result was available for plotting.")
    plot_combined_scatter(runs, writer, args.prediction)

    current_metrics = pd.DataFrame([metric_row(run) for run in runs.values()])
    writer.write_csv(current_metrics, Path("tables") / "selected_proposed_test_metrics.csv")

    traditional = load_traditional_baselines(project_dir, runs)
    if not traditional.empty:
        writer.write_csv(traditional, Path("tables") / "traditional_baseline_comparison_current.csv")
        plot_model_comparison(
            traditional,
            writer,
            Path("comparisons") / "traditional_baselines",
            "Proposed model and traditional baselines",
        )

    graph_metrics, graph_root = collect_graph_metrics(project_dir)
    graph_summary = summarise_repeated_metrics(graph_metrics)
    completeness = graph_completeness(graph_metrics)
    writer.write_csv(completeness, Path("tables") / "graph_benchmark_completeness.csv")
    if not graph_metrics.empty:
        writer.write_csv(graph_metrics, Path("tables") / "graph_benchmark_per_seed_available.csv")
        writer.write_csv(graph_summary, Path("tables") / "graph_benchmark_mean_std_available.csv")
        plot_model_comparison(
            graph_summary,
            writer,
            Path("comparisons") / "graph_benchmarks",
            "Available graph-model benchmark results",
            use_errorbars=True,
        )
    else:
        warn("No completed graph-model test metrics were available for comparison plots.")
    plot_external_training_histories(project_dir, writer)

    ablations = collect_ablation_metrics(project_dir)
    if not ablations.empty:
        writer.write_csv(ablations, Path("tables") / "ablation_metrics_available.csv")
        plot_ablations(ablations, writer)

    write_text_report(writer, runs, graph_metrics, completeness, ablations)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "project_dir": str(project_dir),
        "output_dir": str(output_dir),
        "prediction_variant_requested": args.prediction,
        "selected_runs": {
            protocol: {
                "results_csv": str(run.source.resolve()),
                "seed": run.seed,
                "prediction_column": run.prediction_column,
                "source_unit": run.source_unit,
                "split_counts": run.split_counts,
            }
            for protocol, run in runs.items()
        },
        "graph_benchmark_root": str(graph_root.resolve()) if graph_root is not None else None,
        "graph_test_metric_runs_available": int(len(graph_metrics)),
        "graph_expected_run_count": int(len(completeness)),
        "ablation_summary_rows_available": int(len(ablations)),
        "warnings": WARNINGS,
        "generated_files": writer.generated,
    }
    manifest_path = output_dir / "plot_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {manifest_path}")
    print(f"Finished. Figures and tables are in: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
