#!/usr/bin/env python
"""Fixed-cage-size physical analysis of fullerene HOMO-LUMO gaps.

This post-processing script is aligned with the global structural descriptors
used in gap2.py. It reads an existing DFT/property table, evaluates
descriptor-gap associations separately within fixed cage sizes, and generates
a three-panel publication figure:

  (a) fixed-N Spearman correlation heatmap;
  (b) descriptor-wise median correlation with bootstrap confidence intervals;
  (c) a representative fixed-N descriptor-gap scatter plot.

The default analysis focuses on C50, C52, C54, C56, C58, and C60 so that cage
size is controlled while retaining well-populated isomer groups. The
descriptors are restricted to the quantities actually used by the model:
three adjacency-spectrum descriptors, six 3D shape descriptors, and pentagon
adjacency.

No DFT calculation and no machine-learning training are performed. If the
input CSV does not contain all ten descriptors, the companion extractor reads
the matched Gaussian geometries, generates a complete descriptor table using
the same definitions as gap2.py, and uses that table for the analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HARTREE_TO_EV = 27.211386245988
SCRIPT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA = SCRIPT_DIR / "data" / "dataset.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "physical_analysis"
DEFAULT_DESCRIPTOR_DATA = DEFAULT_OUTPUT_DIR / "fullerene_physical_descriptors.csv"
LOCAL_GEOMETRY_ROOT = SCRIPT_DIR / "data" / "structures"
FALLBACK_GEOMETRY_ROOT = LOCAL_GEOMETRY_ROOT


@dataclass(frozen=True)
class DescriptorSpec:
    key: str
    label: str
    category: str
    aliases: tuple[str, ...]


# Descriptor set aligned with calculate_comprehensive_physics() and
# MatrixFullereneFeature in gap2.py. The first matching alias is used.
#
# phys_feat order in gap2.py:
#   0 normalized lower-half adjacency-eigenvalue sum
#   1 adjacency-eigenvalue standard deviation
#   2 central adjacency spectral gap
#   3 mean radial distance
#   4 radial-distance standard deviation
#   5 asphericity
#   6 convex-hull volume / N
#   7 convex-hull area / N^(2/3)
#   8 isoperimetric deviation
DIRECT_DESCRIPTOR_SPECS = (
    DescriptorSpec(
        "mean_radius",
        r"Mean radial distance ($\mathrm{\AA}$)",
        "geometry",
        (
            "mean_radius",
            "mean_radial_distance",
            "radial_mean",
            "radius_mean",
            "Geo_Radial_Mean",
            "phys_feat_3",
            "phys_3",
        ),
    ),
    DescriptorSpec(
        "radial_std",
        r"Radial-distance std ($\mathrm{\AA}$)",
        "geometry",
        (
            "radial_std",
            "radial_distortion",
            "radius_std",
            "Geo_Radial_Std",
            "phys_feat_4",
            "phys_4",
        ),
    ),
    DescriptorSpec(
        "asphericity",
        "Asphericity",
        "geometry",
        (
            "asphericity",
            "Geo_F_asym",
            "Geo_Asphericity",
            "phys_feat_5",
            "phys_5",
        ),
    ),
    DescriptorSpec(
        "volume_per_atom",
        r"$V/N$ ($\mathrm{\AA}^3$)",
        "geometry",
        (
            "volume_per_atom",
            "volume_norm",
            "Geo_Volume_per_atom",
            "Geo_Volume_norm",
            "phys_feat_6",
            "phys_6",
        ),
    ),
    DescriptorSpec(
        "area_scaled",
        r"$A/N^{2/3}$ ($\mathrm{\AA}^2$)",
        "geometry",
        (
            "area_scaled",
            "area_norm",
            "area_over_n_2_3",
            "Geo_Area_scaled",
            "Geo_Area_norm",
            "phys_feat_7",
            "phys_7",
        ),
    ),
    DescriptorSpec(
        "isoperimetric_deviation",
        "Isoperimetric deviation",
        "geometry",
        (
            "isoperimetric_deviation",
            "d_ipq",
            "D_IPQ",
            "d_ipo",
            "Geo_D_IPQ",
            "Geo_D_IPO",
            "phys_feat_8",
            "phys_8",
        ),
    ),
    DescriptorSpec(
        "pentagon_adjacency",
        "Pentagon adjacency",
        "topology",
        (
            "pentagon_adjacency",
            "adjacent_pentagon_count",
            "adj_pentagons",
            "ipr_penalty",
            "IPR_Penalty",
            "Phys_P5_P5_Count",
        ),
    ),
    DescriptorSpec(
        "spectral_lower_sum",
        "Normalized lower spectral sum",
        "topology",
        (
            "spectral_lower_sum",
            "adjacency_lower_spectral_sum",
            "normalized_lower_spectral_sum",
            "eig_lower_sum_norm",
            "phys_feat_0",
            "phys_0",
        ),
    ),
    DescriptorSpec(
        "spectral_std",
        "Adjacency spectral std",
        "topology",
        (
            "spectral_std",
            "adjacency_spectral_std",
            "eigenvalue_std",
            "eig_std",
            "phys_feat_1",
            "phys_1",
        ),
    ),
    DescriptorSpec(
        "central_spectral_gap",
        "Central adjacency spectral gap",
        "topology",
        (
            "central_spectral_gap",
            "adjacency_central_gap",
            "spectral_gap",
            "eig_central_gap",
            "phys_feat_2",
            "phys_2",
        ),
    ),
)

# These two descriptors may be reconstructed from extensive convex-hull
# quantities if the normalized columns are absent. The normalization exactly
# matches gap2.py: volume/N and area/N^(2/3).
DERIVED_GEOMETRY_SPECS = (
    (
        "volume_per_atom",
        ("Geo_Volume", "volume", "Volume", "convex_hull_volume"),
        1.0,
        "{column}/N",
    ),
    (
        "area_scaled",
        ("Geo_Area", "area", "Area", "convex_hull_area"),
        2.0 / 3.0,
        "{column}/N^(2/3)",
    ),
)

DEFAULT_FIXED_SIZES = (50, 52, 54, 56, 58, 60)


def configure_style() -> None:
    """Apply a journal-ready Arial style."""
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.bfit": "Arial:italic:bold",
            "mathtext.sf": "Arial",
            "mathtext.tt": "Arial",
            "mathtext.cal": "Arial:italic",
            "mathtext.fallback": None,
            "mathtext.default": "regular",
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "axes.linewidth": 1.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "savefig.bbox": "tight",
        }
    )


def enforce_arial(fig: plt.Figure) -> None:
    """Force Arial on every Matplotlib text artist, including colorbars."""
    for text_artist in fig.findobj(match=mpl.text.Text):
        text_artist.set_fontfamily("Arial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze structure-dependent fullerene gap variation within fixed "
            "cage sizes; no model retraining is performed."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--descriptor-data", type=Path, default=DEFAULT_DESCRIPTOR_DATA,
        help="Complete ten-descriptor CSV generated from Gaussian geometries.",
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=(
            LOCAL_GEOMETRY_ROOT
            if LOCAL_GEOMETRY_ROOT.is_dir()
            else FALLBACK_GEOMETRY_ROOT
        ),
        help="Root of Organized_Fullerenes used when extraction is required.",
    )
    parser.add_argument(
        "--rebuild-descriptors",
        action="store_true",
        help="Recompute all ten descriptors even if the enriched CSV exists.",
    )
    parser.add_argument(
        "--extraction-workers",
        type=int,
        default=1,
        help="Number of processes used for descriptor extraction (default: 1).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-size-count",
        type=int,
        default=20,
        help="Minimum number of isomers required for a cage size (default: 20).",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_FIXED_SIZES),
        help=(
            "Cage sizes used for the fixed-N analysis. "
            "Default: 50 52 54 56 58 60."
        ),
    )
    parser.add_argument(
        "--all-eligible-sizes",
        action="store_true",
        help=(
            "Ignore --sizes and use every cage size with at least "
            "--min-size-count samples."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of size-level bootstrap resamples for panel (b).",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--gap-unit",
        choices=("auto", "Hartree", "eV"),
        default="auto",
        help="Unit of the input gap column. Auto detects the current table.",
    )
    parser.add_argument(
        "--scatter-descriptor",
        default=None,
        help="Optional descriptor key for panel (c), e.g. asphericity.",
    )
    parser.add_argument(
        "--scatter-size",
        type=int,
        default=None,
        help="Optional cage size for panel (c).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=1200,
        help="Raster output resolution in dots per inch (default: 1200).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
    )
    return parser.parse_args()


def resolve_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        match = lookup.get(str(alias).strip().lower())
        if match is not None:
            return match
    return None


def table_has_all_descriptors(data_path: Path) -> bool:
    """Check descriptor coverage from the CSV header without loading all rows."""
    if not data_path.is_file():
        return False
    columns = pd.read_csv(data_path, nrows=0).columns
    return all(
        resolve_column(columns, spec.aliases) is not None
        for spec in DIRECT_DESCRIPTOR_SPECS
    )


def ensure_complete_descriptor_data(args: argparse.Namespace) -> Path:
    """Return a complete table, extracting it from .gjf files when necessary."""
    descriptor_data = args.descriptor_data.resolve()
    if not args.rebuild_descriptors and table_has_all_descriptors(descriptor_data):
        print(f"Using complete descriptor table: {descriptor_data}")
        return descriptor_data

    if not args.rebuild_descriptors and table_has_all_descriptors(args.data):
        print(f"Input table already contains all ten descriptors: {args.data}")
        return args.data.resolve()

    try:
        from scripts.analysis.extract_physical_descriptors import build_descriptor_table
    except ImportError as error:
        raise ImportError(
            "extract_physical_descriptors.py must be located beside this script"
        ) from error

    print(
        "The available table does not contain all ten gap2.py-aligned "
        "descriptors; starting geometry-based extraction."
    )
    return build_descriptor_table(
        args.data,
        args.geometry_root,
        descriptor_data,
        workers=args.extraction_workers,
    )


def add_descriptor(
    work: pd.DataFrame,
    metadata: list[dict[str, str]],
    key: str,
    label: str,
    category: str,
    values: pd.Series,
    source: str,
) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < 3 or numeric.nunique(dropna=True) < 2:
        return
    work[key] = numeric.astype(float)
    metadata.append(
        {
            "key": key,
            "label": label,
            "category": category,
            "source": source,
        }
    )


def prepare_data(
    data_path: Path, gap_unit: str
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Input data file was not found: {data_path}")

    raw = pd.read_csv(data_path)
    n_column = resolve_column(raw.columns, ("N", "AtomCount", "atom_size", "num_atoms"))
    gap_column = resolve_column(
        raw.columns,
        ("DFT_gap", "Gap", "gap", "HOMO_LUMO_gap", "target"),
    )
    if n_column is None or gap_column is None:
        raise ValueError(
            "The input table must contain a cage-size column (N or AtomCount) "
            "and a DFT gap column (DFT_gap or Gap)."
        )

    work = pd.DataFrame(
        {
            "N": pd.to_numeric(raw[n_column], errors="coerce"),
            "gap_input": pd.to_numeric(raw[gap_column], errors="coerce"),
        }
    )
    work = work.dropna(subset=["N", "gap_input"]).copy()
    work["N"] = work["N"].astype(int)

    detected_unit = gap_unit
    if gap_unit == "auto":
        finite_gap = work["gap_input"].replace([np.inf, -np.inf], np.nan).dropna()
        if finite_gap.empty:
            raise ValueError("The gap column contains no finite values.")
        detected_unit = "Hartree" if finite_gap.abs().quantile(0.99) < 0.5 else "eV"

    if detected_unit == "Hartree":
        work["gap_eV"] = work["gap_input"] * HARTREE_TO_EV
    else:
        work["gap_eV"] = work["gap_input"]

    metadata: list[dict[str, str]] = []

    # Load direct columns first.
    loaded_keys: set[str] = set()
    spec_by_key = {spec.key: spec for spec in DIRECT_DESCRIPTOR_SPECS}
    for spec in DIRECT_DESCRIPTOR_SPECS:
        source_column = resolve_column(raw.columns, spec.aliases)
        if source_column is None:
            continue
        add_descriptor(
            work,
            metadata,
            spec.key,
            spec.label,
            spec.category,
            raw.loc[work.index, source_column],
            source_column,
        )
        if spec.key in work.columns:
            loaded_keys.add(spec.key)

    # Reconstruct V/N and A/N^(2/3) only when their normalized columns
    # are not already available.
    for key, total_aliases, exponent, source_template in DERIVED_GEOMETRY_SPECS:
        if key in loaded_keys:
            continue
        total_column = resolve_column(raw.columns, total_aliases)
        if total_column is None:
            continue
        spec = spec_by_key[key]
        total_values = pd.to_numeric(
            raw.loc[work.index, total_column], errors="coerce"
        )
        denominator = work["N"].astype(float) ** exponent
        values = total_values / denominator
        add_descriptor(
            work,
            metadata,
            spec.key,
            spec.label,
            spec.category,
            values,
            source_template.format(column=total_column),
        )
        if spec.key in work.columns:
            loaded_keys.add(spec.key)

    metadata_frame = pd.DataFrame(metadata)
    if metadata_frame.empty:
        expected = ", ".join(spec.key for spec in DIRECT_DESCRIPTOR_SPECS)
        raise ValueError(
            "No model-aligned descriptor columns were found in the input table. "
            "Expected one or more of: "
            f"{expected}. If sorted_fullerene_data.csv contains only labels and "
            "identifiers, first generate a descriptor-enriched CSV using the "
            "same definitions as gap2.py."
        )

    # Preserve the model-defined descriptor order in tables/figures.
    order = {spec.key: i for i, spec in enumerate(DIRECT_DESCRIPTOR_SPECS)}
    metadata_frame["descriptor_order"] = metadata_frame["key"].map(order)
    metadata_frame = (
        metadata_frame.sort_values("descriptor_order")
        .drop(columns="descriptor_order")
        .reset_index(drop=True)
    )

    work = work.replace([np.inf, -np.inf], np.nan)
    return work, metadata_frame, detected_unit


def spearman_rho(
    x: pd.Series, y: pd.Series
) -> tuple[float, float, int, int]:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    valid_n = len(pair)
    unique_n = pair["x"].nunique()
    if valid_n < 3 or unique_n < 2 or pair["y"].nunique() < 2:
        return np.nan, np.nan, valid_n, unique_n

    # A three-regular graph has an adjacency-spectrum standard deviation of
    # sqrt(3).  Numerical eigensolvers can nevertheless introduce variations
    # around 1e-16, which must not be treated as physical rank information.
    x_values = pair["x"].to_numpy(dtype=float)
    y_values = pair["y"].to_numpy(dtype=float)
    x_scale = max(1.0, float(np.max(np.abs(x_values))))
    y_scale = max(1.0, float(np.max(np.abs(y_values))))
    if np.ptp(x_values) <= 1e-12 * x_scale:
        return np.nan, np.nan, valid_n, 1
    if np.ptp(y_values) <= 1e-12 * y_scale:
        return np.nan, np.nan, valid_n, unique_n

    result = spearmanr(x_values, y_values)
    return float(result.statistic), float(result.pvalue), valid_n, unique_n


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction; NaN values remain NaN."""
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if finite_idx.size == 0:
        return pd.Series(q, index=p_values.index, dtype=float)

    finite_p = p[finite_idx]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    m = ranked.size
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    original_order_adjusted = np.empty_like(adjusted)
    original_order_adjusted[order] = adjusted
    q[finite_idx] = original_order_adjusted
    return pd.Series(q, index=p_values.index, dtype=float)


def calculate_fixed_size_correlations(
    work: pd.DataFrame,
    metadata: pd.DataFrame,
    min_size_count: int,
    requested_sizes: list[int],
    all_eligible_sizes: bool,
) -> tuple[pd.DataFrame, list[int]]:
    counts = work.groupby("N").size()

    if all_eligible_sizes:
        eligible_sizes = sorted(
            counts[counts >= min_size_count].index.astype(int).tolist()
        )
    else:
        requested_unique = list(dict.fromkeys(int(size) for size in requested_sizes))
        eligible_sizes = [
            size
            for size in requested_unique
            if int(counts.get(size, 0)) >= min_size_count
        ]
        missing_or_sparse = [
            size
            for size in requested_unique
            if int(counts.get(size, 0)) < min_size_count
        ]
        if missing_or_sparse:
            details = ", ".join(
                f"C{size} (n={int(counts.get(size, 0))})"
                for size in missing_or_sparse
            )
            print(
                "Warning: requested fixed-N groups below --min-size-count "
                f"were skipped: {details}"
            )

    if not eligible_sizes:
        raise ValueError(
            "No selected cage size satisfies the minimum sample-count criterion."
        )

    records: list[dict[str, object]] = []
    for cage_size in eligible_sizes:
        group = work[work["N"].eq(cage_size)]
        for descriptor in metadata.itertuples(index=False):
            rho, p_value, valid_n, unique_n = spearman_rho(
                group[descriptor.key], group["gap_eV"]
            )
            records.append(
                {
                    "N": cage_size,
                    "size_sample_count": len(group),
                    "descriptor_key": descriptor.key,
                    "descriptor_label": descriptor.label,
                    "category": descriptor.category,
                    "rho": rho,
                    "p_value": p_value,
                    "valid_pair_count": valid_n,
                    "unique_descriptor_values": unique_n,
                }
            )

    result = pd.DataFrame(records)
    result["q_value_bh"] = np.nan
    for _, idx in result.groupby("N").groups.items():
        result.loc[idx, "q_value_bh"] = benjamini_hochberg(
            result.loc[idx, "p_value"]
        ).to_numpy()
    return result, eligible_sizes


def bootstrap_median_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    median = float(np.median(values))
    if values.size == 1 or n_bootstrap <= 0:
        return median, median, median
    sample_indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    bootstrap_medians = np.median(values[sample_indices], axis=1)
    lower, upper = np.quantile(bootstrap_medians, (0.025, 0.975))
    return median, float(lower), float(upper)


def summarize_correlations(
    correlation_long: pd.DataFrame,
    metadata: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for descriptor in metadata.itertuples(index=False):
        values = correlation_long.loc[
            correlation_long["descriptor_key"].eq(descriptor.key), "rho"
        ].dropna().to_numpy(dtype=float)
        median, lower, upper = bootstrap_median_ci(values, rng, n_bootstrap)
        nonzero = values[values != 0]
        if nonzero.size:
            sign_consistency = max(
                float(np.mean(nonzero > 0)),
                float(np.mean(nonzero < 0)),
            )
        else:
            sign_consistency = np.nan
        records.append(
            {
                "descriptor_key": descriptor.key,
                "descriptor_label": descriptor.label,
                "category": descriptor.category,
                "median_rho": median,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "valid_size_count": int(values.size),
                "sign_consistency": sign_consistency,
            }
        )
    return pd.DataFrame(records)


def choose_scatter_descriptor(
    summary: pd.DataFrame,
    requested: str | None,
) -> str:
    if requested is not None:
        requested_lower = requested.strip().lower()
        match = summary[
            summary["descriptor_key"].str.lower().eq(requested_lower)
            | summary["descriptor_label"].str.lower().eq(requested_lower)
        ]
        if match.empty:
            available = ", ".join(summary["descriptor_key"])
            raise ValueError(
                f"Unknown scatter descriptor '{requested}'. Available: {available}"
            )
        return str(match.iloc[0]["descriptor_key"])

    # Select objectively from the complete model-aligned descriptor set.  The
    # earlier geometry-only restriction could hide a stronger and more stable
    # topology-gap relationship once the spectral descriptors were extracted.
    candidates = summary[
        summary["median_rho"].notna()
        & summary["sign_consistency"].notna()
    ].copy()
    candidates["selection_score"] = (
        candidates["median_rho"].abs()
        * candidates["sign_consistency"]
        * np.sqrt(candidates["valid_size_count"].clip(lower=1))
    )
    return str(
        candidates.sort_values(
            ["selection_score", "valid_size_count"], ascending=False
        ).iloc[0]["descriptor_key"]
    )


def choose_scatter_size(
    work: pd.DataFrame,
    correlation_long: pd.DataFrame,
    descriptor_key: str,
    requested: int | None,
) -> int:
    available = correlation_long[
        correlation_long["descriptor_key"].eq(descriptor_key)
        & correlation_long["rho"].notna()
    ].copy()
    if requested is not None:
        if requested not in set(available["N"].astype(int)):
            raise ValueError(
                f"C{requested} is not eligible for descriptor '{descriptor_key}'."
            )
        return int(requested)

    # Select a well-populated size whose correlation is representative of the
    # descriptor's cross-size median. This avoids cherry-picking the largest
    # observed correlation while also avoiding a large but atypical size.
    size_counts = work.groupby("N").size().rename("sample_count")
    available = available.join(size_counts, on="N")
    median_rho = float(available["rho"].median())
    available["median_distance"] = (available["rho"] - median_rho).abs()
    return int(
        available.sort_values(
            ["median_distance", "sample_count", "N"],
            ascending=[True, False, True],
        ).iloc[0]["N"]
    )


def draw_heatmap(
    ax: plt.Axes,
    correlation_long: pd.DataFrame,
    metadata: pd.DataFrame,
    eligible_sizes: list[int],
) -> mpl.image.AxesImage:
    keys = metadata["key"].tolist()
    labels = metadata["label"].tolist()
    matrix = (
        correlation_long.pivot(index="N", columns="descriptor_key", values="rho")
        .reindex(index=eligible_sizes, columns=keys)
        .to_numpy(dtype=float)
    )
    q_matrix = (
        correlation_long.pivot(
            index="N", columns="descriptor_key", values="q_value_bh"
        )
        .reindex(index=eligible_sizes, columns=keys)
        .to_numpy(dtype=float)
    )
    cmap = mpl.colormaps["coolwarm"].copy()
    cmap.set_bad("#EEEEEE")
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontfamily="Arial")
    ax.set_yticks(np.arange(len(eligible_sizes)))
    ax.set_yticklabels(
        [rf"C$_{{{size}}}$" for size in eligible_sizes], fontfamily="Arial"
    )
    ax.set_xlabel("Structural descriptor", fontweight="bold")
    ax.set_ylabel("Cage size", fontweight="bold")
    ax.set_title("(a) Fixed-size descriptor-gap correlations", fontweight="bold")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            rho = matrix[row_index, column_index]
            q_value = q_matrix[row_index, column_index]
            if np.isfinite(rho) and np.isfinite(q_value) and q_value < 0.05:
                star_color = "white" if abs(rho) >= 0.55 else "black"
                ax.text(
                    column_index,
                    row_index,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color=star_color,
                    fontfamily="Arial",
                )
    return image


def draw_summary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    # Group descriptors by scientific role rather than correlation magnitude.
    # This fixed order makes the topology/geometry comparison immediately clear.
    topology_order = [
        "central_spectral_gap",
        "spectral_lower_sum",
        "pentagon_adjacency",
        "spectral_std",
    ]
    geometry_order = [
        "mean_radius",
        "radial_std",
        "asphericity",
        "isoperimetric_deviation",
        "volume_per_atom",
        "area_scaled",
    ]
    desired_order = topology_order + geometry_order
    ordered = (
        summary.set_index("descriptor_key")
        .reindex(desired_order)
        .dropna(subset=["descriptor_label"])
        .reset_index()
    )
    topology_count = int(ordered["category"].eq("topology").sum())
    geometry_count = int(ordered["category"].eq("geometry").sum())
    y = np.concatenate(
        [
            np.arange(topology_count, dtype=float),
            np.arange(geometry_count, dtype=float) + topology_count + 1.0,
        ]
    )
    colors = [
        "#2A9D8F" if category == "geometry" else "#E76F51"
        for category in ordered["category"]
    ]
    medians = ordered["median_rho"].to_numpy(float)
    lower = ordered["ci95_lower"].to_numpy(float)
    upper = ordered["ci95_upper"].to_numpy(float)
    finite = np.isfinite(medians) & np.isfinite(lower) & np.isfinite(upper)
    xerr = np.vstack(
        [medians[finite] - lower[finite], upper[finite] - medians[finite]]
    )
    ax.errorbar(
        medians[finite],
        y[finite],
        xerr=xerr,
        fmt="none",
        ecolor="#444444",
        elinewidth=1.8,
        capsize=6,
        capthick=1.5,
        zorder=1,
    )
    finite_colors = [color for color, keep in zip(colors, finite) if keep]
    ax.scatter(
        medians[finite],
        y[finite],
        c=finite_colors,
        s=55,
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )
    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1.1)
    separator_y = topology_count - 0.5 + 0.5
    ax.axhline(separator_y, color="#C8C8C8", linewidth=1.0, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["descriptor_label"], fontfamily="Arial")
    ax.set_ylim(y[-1] + 0.55, -0.95)
    finite_limits = np.concatenate([lower[np.isfinite(lower)], upper[np.isfinite(upper)]])
    limit = 0.25 if finite_limits.size == 0 else max(0.25, np.max(np.abs(finite_limits)) * 1.15)
    ax.set_xlim(-min(limit, 1.0), min(limit, 1.0))
    ax.set_xlabel(r"Median fixed-size Spearman $\rho$", fontweight="bold")
    ax.set_title("(b) Cross-size correlation summary", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7, alpha=0.7)
    ax.text(
        0.015,
        -0.48,
        "Topology / spectral",
        transform=ax.get_yaxis_transform(),
        color="#E76F51",
        fontsize=11.5,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        0.015,
        topology_count + 0.48,
        "Geometry",
        transform=ax.get_yaxis_transform(),
        color="#2A9D8F",
        fontsize=11.5,
        fontweight="bold",
        ha="left",
        va="center",
    )


def binned_median_trend(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    bin_count = min(10, max(4, len(x_sorted) // 60))
    groups = np.array_split(np.arange(len(x_sorted)), bin_count)
    median_x = np.array([np.median(x_sorted[group]) for group in groups])
    median_y = np.array([np.median(y_sorted[group]) for group in groups])
    return median_x, median_y


def draw_representative_scatter(
    ax: plt.Axes,
    work: pd.DataFrame,
    metadata: pd.DataFrame,
    descriptor_key: str,
    cage_size: int,
) -> tuple[int, float]:
    descriptor_row = metadata[metadata["key"].eq(descriptor_key)].iloc[0]
    subset = work.loc[
        work["N"].eq(cage_size), [descriptor_key, "gap_eV"]
    ].dropna()
    rho, _, valid_n, _ = spearman_rho(subset[descriptor_key], subset["gap_eV"])
    x = subset[descriptor_key].to_numpy(float)
    y = subset["gap_eV"].to_numpy(float)
    ax.scatter(
        x,
        y,
        s=18,
        color="#3B75AF",
        alpha=0.58,
        edgecolor="white",
        linewidth=0.25,
        rasterized=True,
    )
    trend_x, trend_y = binned_median_trend(x, y)
    ax.plot(
        trend_x,
        trend_y,
        color="#E76F51",
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="Binned median",
    )
    ax.set_xlabel(str(descriptor_row["label"]), fontweight="bold")
    ax.set_ylabel("DFT HOMO-LUMO gap (eV)", fontweight="bold")
    ax.set_title("(c) Representative fixed-size relationship", fontweight="bold")
    ax.text(
        0.04,
        0.95,
        rf"C$_{{{cage_size}}}$, $n={valid_n}$"
        "\n"
        rf"Spearman $\rho={rho:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#BBBBBB"},
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(color="#DDDDDD", linewidth=0.7, alpha=0.65)
    return valid_n, rho


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Fixed_Size_Physical_Gap_Analysis"
    saved: list[Path] = []
    for file_format in formats:
        path = stem.with_suffix(f".{file_format}")
        kwargs: dict[str, object] = {"bbox_inches": "tight"}
        if file_format.lower() == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
        print(f"Saved: {path}")
    return saved


def main() -> None:
    args = parse_args()
    if args.min_size_count < 3:
        raise ValueError("--min-size-count must be at least 3.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative.")
    if args.extraction_workers < 1:
        raise ValueError("--extraction-workers must be at least 1.")

    configure_style()
    analysis_data = ensure_complete_descriptor_data(args)
    work, metadata, detected_unit = prepare_data(analysis_data, args.gap_unit)
    correlation_long, eligible_sizes = calculate_fixed_size_correlations(
        work,
        metadata,
        args.min_size_count,
        args.sizes,
        args.all_eligible_sizes,
    )
    summary = summarize_correlations(
        correlation_long,
        metadata,
        args.bootstrap_samples,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    descriptor_columns = metadata["key"].tolist()
    analysis_values = work.loc[
        work["N"].isin(eligible_sizes),
        ["N", "gap_eV", *descriptor_columns],
    ].copy()
    analysis_values.to_csv(
        args.output_dir / "fixed_N_descriptor_values.csv", index=False
    )

    metadata.to_csv(args.output_dir / "descriptor_sources.csv", index=False)
    correlation_long.to_csv(
        args.output_dir / "fixed_N_spearman_correlations.csv", index=False
    )
    summary.to_csv(
        args.output_dir / "descriptor_correlation_summary.csv", index=False
    )

    fig = plt.figure(figsize=(17.5, 6.8))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.0),
        wspace=0.34,
    )
    heatmap_ax = fig.add_subplot(grid[0])
    summary_ax = fig.add_subplot(grid[1])

    heatmap = draw_heatmap(
        heatmap_ax, correlation_long, metadata, eligible_sizes
    )
    colorbar = fig.colorbar(
        heatmap,
        ax=heatmap_ax,
        fraction=0.045,
        pad=0.025,
    )
    colorbar.set_label("")
    colorbar.ax.set_title(r"$\rho$", fontsize=15, fontweight="bold", pad=8)
    draw_summary(summary_ax, summary)
    fig.suptitle(
        "Fixed-size structural associations with fullerene HOMO-LUMO gaps",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.23, top=0.88)
    enforce_arial(fig)
    save_figure(fig, args.output_dir, args.formats, args.dpi)
    plt.close(fig)

    print(f"Label input: {args.data}")
    print(f"Descriptor input: {analysis_data}")
    print(f"Input gap unit: {detected_unit}; plotted unit: eV")
    print(f"Eligible cage sizes (n >= {args.min_size_count}): {eligible_sizes}")
    print("Descriptors used:")
    used_keys = set(metadata["key"])
    for row in metadata.itertuples(index=False):
        print(f"  {row.key}: {row.label} [{row.category}] <- {row.source}")

    missing_keys = [
        spec.key for spec in DIRECT_DESCRIPTOR_SPECS if spec.key not in used_keys
    ]
    if missing_keys:
        raise RuntimeError(
            "The analysis table is unexpectedly missing model-aligned "
            "descriptors: " + ", ".join(missing_keys)
        )
    print(
        "Interpret the correlations as within-size associations, not as causal effects."
    )


if __name__ == "__main__":
    main()
