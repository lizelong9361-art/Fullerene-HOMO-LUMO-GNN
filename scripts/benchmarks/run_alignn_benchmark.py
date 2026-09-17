#!/usr/bin/env python3
"""Run official ALIGNN on the validated 7,035-fullerene benchmark.

This controller reconstructs ALIGNN inputs from the benchmark cache and
consumes the exact split manifests used by the other benchmark models.
Outputs go to results/graph_benchmarks/runs/alignn/{protocol}/seed_{seed}/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_SAMPLES = 7035
EXPECTED_SEEDS = (2024, 2025, 2026, 2027, 2028)
EXPECTED_COUNTS = {
    "random": {"train": 5628, "val": 703, "test": 704},
    "group": {"train": 2752, "val": 3017, "test": 1266},
}
SPLIT_ORDER = ("train", "val", "test")
MODEL_NAME = "alignn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validated five-seed ALIGNN benchmark for the fullerene dataset."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Repository root containing results/graph_benchmarks.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Override the results/graph_benchmarks directory.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Override validated_raw_dataset_cache.pt.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for shared XYZ files and per-run ALIGNN inputs.",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=tuple(EXPECTED_COUNTS),
        default=list(EXPECTED_COUNTS),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(EXPECTED_SEEDS)
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--train-command",
        default="train_alignn.py",
        help="ALIGNN CLI executable or command prefix.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Run one task from 0..9: random seeds first, then group seeds.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate data and write all selected inputs without training.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only rebuild metric summaries from completed run folders.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs with complete, validated metrics and predictions.",
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Rebuild shared XYZ files and the shared sample table.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    project_dir = (args.project_dir or script_dir.parents[1]).expanduser().resolve()
    benchmark_dir = (
        args.benchmark_dir or project_dir / "results" / "graph_benchmarks"
    ).expanduser().resolve()
    cache_path = (
        args.cache or benchmark_dir / "validated_raw_dataset_cache.pt"
    ).expanduser().resolve()
    work_dir = (
        args.work_dir or project_dir / "results" / "alignn"
    ).expanduser().resolve()
    return project_dir, benchmark_dir, cache_path, work_dir


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def canonical_id(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def canonical_filename(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    return PurePosixPath(text).name


def scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar, received shape {array.shape}.")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"Non-finite scalar: {result}")
    return result


def integer_scalar(value: Any) -> int:
    number = scalar(value)
    rounded = int(round(number))
    if abs(number - rounded) > 1e-8:
        raise ValueError(f"Expected an integer, received {number}.")
    return rounded


def coordinates(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Coordinates must have shape (N, 3), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Coordinates contain NaN or infinite values.")
    return array


def install_numpy_pickle_compatibility() -> None:
    """Allow NumPy 1.x to read caches serialized by NumPy 2.x.

    NumPy 2 stores some pickle references below ``numpy._core`` whereas NumPy
    1.x exposes the same implementation below ``numpy.core``. ALIGNN commonly
    runs in an older NumPy environment, so register aliases before torch.load.
    This does not change numerical behavior or upgrade the ALIGNN environment.
    """
    try:
        importlib.import_module("numpy._core")
        return
    except ModuleNotFoundError:
        pass

    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
        "numpy._core._methods": "numpy.core._methods",
        "numpy._core.shape_base": "numpy.core.shape_base",
        "numpy._core.function_base": "numpy.core.function_base",
        "numpy._core.getlimits": "numpy.core.getlimits",
        "numpy._core._dtype": "numpy.core._dtype",
        "numpy._core.overrides": "numpy.core.overrides",
    }
    for new_name, old_name in aliases.items():
        try:
            sys.modules.setdefault(new_name, importlib.import_module(old_name))
        except ModuleNotFoundError:
            # Only modules referenced by the cache are required. NumPy minor
            # releases do not necessarily expose every optional module above.
            continue


def cache_data(cache_path: Path) -> list[Any]:
    if not cache_path.is_file():
        raise FileNotFoundError(f"Validated cache was not found: {cache_path}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to read the validated cache. Activate the ALIGNN "
            "conda environment before running this script."
        ) from exc
    install_numpy_pickle_compatibility()
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, (list, tuple)):
        raise TypeError("The validated cache does not contain a list-like 'data' object.")
    if len(data) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLES} validated samples, found {len(data)}."
        )
    return list(data)


def get_item_value(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        if name not in item:
            raise KeyError(f"Validated sample is missing '{name}'.")
        return item[name]
    if hasattr(item, name):
        return getattr(item, name)
    try:
        return item[name]
    except Exception as exc:
        raise KeyError(f"Validated sample is missing '{name}'.") from exc


def dataset_fingerprint(benchmark_dir: Path, cache_path: Path) -> str:
    provenance = benchmark_dir / "dataset_provenance.json"
    if provenance.is_file():
        try:
            value = json.loads(provenance.read_text(encoding="utf-8")).get(
                "dataset_fingerprint"
            )
            if value:
                return str(value)
        except (OSError, json.JSONDecodeError):
            pass
    stat = cache_path.stat()
    token = f"{cache_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def shared_data_is_complete(
    table_path: Path, metadata_path: Path, xyz_dir: Path, fingerprint: str
) -> bool:
    if not (table_path.is_file() and metadata_path.is_file() and xyz_dir.is_dir()):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        table = pd.read_csv(table_path)
    except Exception:
        return False
    required = {"row_index", "sample_id", "N", "target", "unit", "xyz_file"}
    if not required.issubset(table.columns):
        return False
    if len(table) != EXPECTED_SAMPLES or table["row_index"].nunique() != EXPECTED_SAMPLES:
        return False
    if metadata.get("dataset_fingerprint") != fingerprint:
        return False
    if set(table["unit"].astype(str)) != {"eV"}:
        return False
    return all((xyz_dir / name).is_file() for name in table["xyz_file"].astype(str))


def prepare_shared_dataset(
    benchmark_dir: Path,
    cache_path: Path,
    work_dir: Path,
    force: bool,
) -> pd.DataFrame:
    xyz_dir = work_dir / "shared_xyz"
    table_path = work_dir / "shared_samples.csv"
    metadata_path = work_dir / "shared_dataset_metadata.json"
    fingerprint = dataset_fingerprint(benchmark_dir, cache_path)
    if not force and shared_data_is_complete(
        table_path, metadata_path, xyz_dir, fingerprint
    ):
        table = pd.read_csv(table_path)
        table["sample_id"] = table["sample_id"].map(canonical_id)
        print(f"Validated existing shared ALIGNN data: {table_path}", flush=True)
        return table.sort_values("row_index").reset_index(drop=True)

    # Atomic directory creation prevents parallel single-run PBS jobs from
    # rebuilding the same 7,035 XYZ files at the same time.
    lock_dir = work_dir / ".prepare_shared_dataset.lock"
    work_dir.mkdir(parents=True, exist_ok=True)
    wait_started = time.monotonic()
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if not force and shared_data_is_complete(
                table_path, metadata_path, xyz_dir, fingerprint
            ):
                table = pd.read_csv(table_path)
                table["sample_id"] = table["sample_id"].map(canonical_id)
                return table.sort_values("row_index").reset_index(drop=True)
            if time.monotonic() - wait_started > 3600:
                raise TimeoutError(
                    f"Timed out waiting for shared-data preparation lock: {lock_dir}"
                )
            print("Another job is preparing shared ALIGNN data; waiting...", flush=True)
            time.sleep(5)

    try:
        # Recheck after taking the lock because another job may have completed
        # preparation between the first check and lock acquisition.
        if not force and shared_data_is_complete(
            table_path, metadata_path, xyz_dir, fingerprint
        ):
            table = pd.read_csv(table_path)
            table["sample_id"] = table["sample_id"].map(canonical_id)
            return table.sort_values("row_index").reset_index(drop=True)

        print(f"Loading validated cache: {cache_path}", flush=True)
        data = cache_data(cache_path)
        xyz_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        keys: set[tuple[int, str]] = set()
        for row_index, item in enumerate(data):
            intended_n = integer_scalar(get_item_value(item, "intended_N"))
            geometry_n = integer_scalar(get_item_value(item, "N"))
            sample_id = canonical_id(get_item_value(item, "sample_id"))
            target = scalar(get_item_value(item, "y_target"))
            xyz = coordinates(get_item_value(item, "coords"))
            has_unit = (isinstance(item, dict) and "unit" in item) or hasattr(item, "unit")
            unit = str(get_item_value(item, "unit")) if has_unit else "eV"
            if unit.lower() != "ev":
                raise RuntimeError(
                    f"Row {row_index} uses unit {unit!r}; the clean benchmark requires eV."
                )
            if intended_n != geometry_n or len(xyz) != intended_n:
                raise RuntimeError(
                    f"Row {row_index} has label N={intended_n}, geometry N={geometry_n}, "
                    f"coordinate rows={len(xyz)}."
                )
            key = (intended_n, sample_id)
            if key in keys:
                raise RuntimeError(f"Duplicate validated sample key: {key}")
            keys.add(key)
            filename = f"row_{row_index:05d}_C{intended_n}_ID{sample_id}.xyz"
            path = xyz_dir / filename
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(f"{intended_n}\n")
                handle.write(
                    f"row_index={row_index} AtomCount={intended_n} ID={sample_id} "
                    f"target_eV={target:.12g}\n"
                )
                for x, y, z in xyz:
                    handle.write(f"C {x:.10f} {y:.10f} {z:.10f}\n")
            rows.append(
                {
                    "row_index": row_index,
                    "sample_id": sample_id,
                    "N": intended_n,
                    "target": target,
                    "unit": "eV",
                    "xyz_file": filename,
                }
            )
            if (row_index + 1) % 1000 == 0:
                print(
                    f"Prepared {row_index + 1}/{EXPECTED_SAMPLES} XYZ files", flush=True
                )

        table = pd.DataFrame(rows)
        atomic_csv(table, table_path)
        atomic_json(
            metadata_path,
            {
                "dataset_fingerprint": fingerprint,
                "validated_cache": str(cache_path),
                "sample_count": len(table),
                "unit": "eV",
                "legacy_alignn_training_data_used": False,
            },
        )
        print(f"Prepared validated shared ALIGNN data: {table_path}", flush=True)
        return table
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


def validate_manifest(
    manifest_path: Path, protocol: str, seed: int, shared: pd.DataFrame
) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest was not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    required = {
        "row_index",
        "sample_id",
        "intended_N",
        "geometry_N",
        "split",
        "protocol",
        "seed",
        "unit",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Manifest {manifest_path} is missing columns: {missing}")
    if len(frame) != EXPECTED_SAMPLES:
        raise RuntimeError(f"Manifest {manifest_path} has {len(frame)} rows, not 7,035.")
    frame = frame.copy()
    frame["row_index"] = pd.to_numeric(frame["row_index"], errors="raise").astype(int)
    frame["split"] = frame["split"].astype(str).str.lower()
    if frame["row_index"].nunique() != EXPECTED_SAMPLES or set(frame["row_index"]) != set(
        range(EXPECTED_SAMPLES)
    ):
        raise RuntimeError(f"Manifest {manifest_path} does not cover each cache row exactly once.")
    if set(frame["protocol"].astype(str).str.lower()) != {protocol}:
        raise RuntimeError(f"Protocol mismatch in {manifest_path}.")
    if set(pd.to_numeric(frame["seed"], errors="raise").astype(int)) != {seed}:
        raise RuntimeError(f"Seed mismatch in {manifest_path}.")
    if set(frame["unit"].astype(str).str.lower()) != {"ev"}:
        raise RuntimeError(f"Unit mismatch in {manifest_path}; expected eV.")
    counts = frame["split"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS[protocol]:
        raise RuntimeError(
            f"Incorrect {protocol} split counts in {manifest_path}: {counts}; "
            f"expected {EXPECTED_COUNTS[protocol]}."
        )
    indexed = shared.set_index("row_index", verify_integrity=True)
    expected_n = frame["row_index"].map(indexed["N"]).astype(int)
    manifest_n = pd.to_numeric(frame["intended_N"], errors="raise").astype(int)
    geometry_n = pd.to_numeric(frame["geometry_N"], errors="raise").astype(int)
    if not (manifest_n.to_numpy() == geometry_n.to_numpy()).all():
        raise RuntimeError(f"Label/geometry atom-count mismatch in {manifest_path}.")
    if not (manifest_n.to_numpy() == expected_n.to_numpy()).all():
        raise RuntimeError(f"Manifest/cache atom-count mismatch in {manifest_path}.")
    manifest_ids = frame["sample_id"].map(canonical_id)
    expected_ids = frame["row_index"].map(indexed["sample_id"]).map(canonical_id)
    if not (manifest_ids.to_numpy() == expected_ids.to_numpy()).all():
        raise RuntimeError(f"Manifest/cache sample-ID mismatch in {manifest_path}.")

    if protocol == "group":
        masks = {
            "train": manifest_n.between(20, 56),
            "val": manifest_n.isin((58, 60)),
            "test": manifest_n.between(70, 100),
        }
        for split, allowed in masks.items():
            actual = frame["split"].eq(split)
            if not actual.equals(allowed):
                raise RuntimeError(
                    f"Strict group membership is incorrect for split={split} in {manifest_path}."
                )
    return frame


def alignn_config(
    seed: int, counts: dict[str, int], output_dir: Path, epochs: int, num_workers: int
) -> dict[str, Any]:
    return {
        "version": "validated_fullerene_five_seed_alignn_v1",
        "dataset": "user_data",
        "target": "target",
        "atom_features": "cgcnn",
        "neighbor_strategy": "k-nearest",
        "id_tag": "jid",
        "dtype": "float32",
        "random_seed": seed,
        "classification_threshold": None,
        "n_train": counts["train"],
        "n_val": counts["val"],
        "n_test": counts["test"],
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "target_multiplication_factor": None,
        "epochs": epochs,
        "batch_size": 32,
        "weight_decay": 1e-5,
        "learning_rate": 0.001,
        # Keep ALIGNN's LMDB cache inside this run folder. An interrupted run
        # is then removed atomically before --resume retrains it, so a partial
        # LMDB can never be mistaken for a complete dataset.
        "filename": str(output_dir / f"gap_{seed}_"),
        "warmup_steps": 2000,
        "criterion": "mse",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "pin_memory": False,
        "save_dataloader": False,
        "write_checkpoint": True,
        "write_predictions": True,
        "store_outputs": True,
        "progress": True,
        "log_tensorboard": False,
        "standard_scalar_and_pca": False,
        "use_canonize": True,
        "num_workers": num_workers,
        "cutoff": 8.0,
        "cutoff_extra": 3.0,
        "max_neighbors": 12,
        "keep_data_order": True,
        "normalize_graph_level_loss": False,
        "distributed": False,
        "data_parallel": False,
        "n_early_stopping": 70,
        "output_dir": str(output_dir),
        # The official non-LMDB dataset path has an interface mismatch in some
        # ALIGNN releases (target_additional_output). The LMDB implementation
        # is the supported path and accepts the complete loader interface.
        "use_lmdb": True,
        "model": {
            "name": "alignn_atomwise",
            "alignn_layers": 4,
            "gcn_layers": 4,
            "atom_input_features": 92,
            "edge_input_features": 80,
            "triplet_input_features": 40,
            "embedding_features": 64,
            "hidden_features": 256,
            "output_features": 1,
            "grad_multiplier": -1,
            "calculate_gradient": False,
            "atomwise_output_features": 0,
            "graphwise_weight": 1.0,
            "gradwise_weight": 0.0,
            "stresswise_weight": 0.0,
            "atomwise_weight": 0.0,
            "link": "identity",
            "zero_inflated": False,
            "classification": False,
            "force_mult_natoms": False,
            "energy_mult_natoms": False,
            "include_pos_deriv": False,
            "use_cutoff_function": False,
            "inner_cutoff": 3.0,
            "stress_multiplier": 1.0,
            "add_reverse_forces": True,
            "lg_on_fly": True,
            "batch_stress": True,
            "multiply_cutoff": False,
            "use_penalty": False,
            "extra_features": 0,
        },
    }


def prepare_run_input(
    protocol: str,
    seed: int,
    benchmark_dir: Path,
    work_dir: Path,
    shared: pd.DataFrame,
    epochs: int,
    num_workers: int,
) -> tuple[Path, Path, Path, pd.DataFrame]:
    manifest_path = benchmark_dir / "manifests" / protocol / f"seed_{seed}.csv"
    manifest = validate_manifest(manifest_path, protocol, seed, shared)
    indexed = shared.set_index("row_index", verify_integrity=True)
    if protocol == "random":
        # Reproduce the exact RandomState permutation used to create the clean
        # benchmark. The manifest is still the authority for membership and is
        # checked against this order before ALIGNN input is written.
        row_order = np.random.RandomState(seed).permutation(EXPECTED_SAMPLES)
        ordered = manifest.set_index("row_index", verify_integrity=True).loc[row_order]
        ordered = ordered.reset_index()
        expected_splits = np.asarray(
            [
                "train",
                "val",
                "test",
            ],
            dtype=object,
        ).repeat(
            [
                EXPECTED_COUNTS[protocol]["train"],
                EXPECTED_COUNTS[protocol]["val"],
                EXPECTED_COUNTS[protocol]["test"],
            ]
        )
        if not np.array_equal(ordered["split"].to_numpy(), expected_splits):
            raise RuntimeError(
                f"Random manifest {manifest_path} does not match RandomState(seed={seed})."
            )
    else:
        parts = [manifest[manifest["split"].eq(split)] for split in SPLIT_ORDER]
        ordered = pd.concat(parts, ignore_index=True)
    ordered["input_order"] = np.arange(len(ordered))
    ordered["sample_id"] = ordered["row_index"].map(indexed["sample_id"])
    ordered["N"] = ordered["row_index"].map(indexed["N"]).astype(int)
    ordered["target"] = ordered["row_index"].map(indexed["target"]).astype(float)
    ordered["unit"] = "eV"
    ordered["xyz_file"] = ordered["row_index"].map(indexed["xyz_file"])

    input_dir = work_dir / "inputs" / protocol / f"seed_{seed}"
    input_dir.mkdir(parents=True, exist_ok=True)
    shared_xyz = work_dir / "shared_xyz"
    with (input_dir / "id_prop.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in ordered.itertuples(index=False):
            xyz_path = shared_xyz / row.xyz_file
            relative = os.path.relpath(xyz_path, input_dir).replace("\\", "/")
            writer.writerow([relative, f"{row.target:.12g}"])
    ordered.to_csv(input_dir / "ordered_manifest.csv", index=False)

    result_dir = benchmark_dir / "runs" / MODEL_NAME / protocol / f"seed_{seed}"
    native_dir = result_dir / "alignn_native"
    counts = EXPECTED_COUNTS[protocol]
    config = alignn_config(seed, counts, native_dir, epochs, num_workers)
    atomic_json(input_dir / "config.json", config)
    atomic_json(
        input_dir / "input_provenance.json",
        {
            "source_manifest": str(manifest_path),
            "shared_samples": str(work_dir / "shared_samples.csv"),
            "protocol": protocol,
            "seed": seed,
            "counts": counts,
            "unit": "eV",
            "ordering": (
                "exact_numpy_randomstate_permutation"
                if protocol == "random"
                else "train_then_validation_then_test_in_cache_order"
            ),
            "legacy_alignn_input_used": False,
        },
    )
    return input_dir, result_dir, native_dir, ordered


def locate_output(root: Path, filename: str) -> Path | None:
    direct = root / filename
    if direct.is_file():
        return direct
    matches = list(root.rglob(filename)) if root.exists() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        matches.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        return matches[0]
    return None


def load_id_splits(native_dir: Path) -> dict[str, list[str]]:
    path = locate_output(native_dir, "ids_train_val_test.json")
    if path is None:
        raise FileNotFoundError("ALIGNN did not write ids_train_val_test.json.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for split in SPLIT_ORDER:
        key = f"id_{split}"
        if key not in payload:
            raise RuntimeError(f"{path} is missing {key}.")
        result[split] = [canonical_filename(value) for value in payload[key]]
    return result


def flatten_json_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected ALIGNN result structure: {path}")
    targets: list[float] = []
    predictions: list[float] = []
    for record in payload:
        if not isinstance(record, dict):
            continue
        target = np.asarray(record.get("target_out", []), dtype=float).reshape(-1)
        prediction = np.asarray(record.get("pred_out", []), dtype=float).reshape(-1)
        if len(target) != len(prediction):
            raise RuntimeError(f"Target/prediction length mismatch in {path}.")
        targets.extend(target.tolist())
        predictions.extend(prediction.tolist())
    return np.asarray(targets, dtype=float), np.asarray(predictions, dtype=float)


def load_split_predictions(
    native_dir: Path, split: str, ids: list[str]
) -> pd.DataFrame:
    csv_path = locate_output(native_dir, f"prediction_results_{split}_set.csv")
    if csv_path is not None:
        frame = pd.read_csv(csv_path)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        if {"target", "prediction"}.issubset(frame.columns) and len(frame) == len(ids):
            output_ids = (
                frame["id"].map(canonical_filename).tolist()
                if "id" in frame.columns
                else ids
            )
            return pd.DataFrame(
                {
                    "xyz_file": output_ids,
                    "native_target": pd.to_numeric(frame["target"], errors="raise"),
                    "pred": pd.to_numeric(frame["prediction"], errors="raise"),
                }
            )

    json_name = {"train": "Train_results.json", "val": "Val_results.json", "test": "Test_results.json"}[split]
    json_path = locate_output(native_dir, json_name)
    if json_path is None:
        raise FileNotFoundError(
            f"No usable ALIGNN prediction output was found for split={split}."
        )
    targets, predictions = flatten_json_predictions(json_path)
    if len(targets) != len(ids):
        raise RuntimeError(
            f"ALIGNN {split} predictions contain {len(targets)} rows; expected {len(ids)}."
        )
    return pd.DataFrame(
        {"xyz_file": ids, "native_target": targets, "pred": predictions}
    )


def validate_and_collect_predictions(
    native_dir: Path,
    ordered: pd.DataFrame,
    protocol: str,
    seed: int,
) -> pd.DataFrame:
    id_splits = load_id_splits(native_dir)
    frames: list[pd.DataFrame] = []
    for split in SPLIT_ORDER:
        expected = ordered[ordered["split"].eq(split)].copy()
        expected_names = expected["xyz_file"].map(canonical_filename).tolist()
        native_ids = id_splits[split]
        if len(native_ids) != EXPECTED_COUNTS[protocol][split]:
            raise RuntimeError(
                f"ALIGNN used {len(native_ids)} {split} samples; expected "
                f"{EXPECTED_COUNTS[protocol][split]}."
            )
        if len(set(native_ids)) != len(native_ids) or set(native_ids) != set(expected_names):
            raise RuntimeError(
                f"ALIGNN's internal {split} IDs do not match the validated manifest."
            )
        try:
            prediction = load_split_predictions(native_dir, split, native_ids)
            expected_by_name = expected.set_index("xyz_file", verify_integrity=True)
            prediction["xyz_file"] = prediction["xyz_file"].map(canonical_filename)
            if len(set(prediction["xyz_file"])) != len(prediction):
                raise RuntimeError(f"Duplicate ALIGNN prediction IDs for split={split}.")
            prediction["true"] = prediction["xyz_file"].map(expected_by_name["target"])
            if prediction["true"].isna().any():
                raise RuntimeError(f"Unknown ALIGNN prediction ID for split={split}.")
            native_target = pd.to_numeric(
                prediction["native_target"], errors="raise"
            ).to_numpy()
            true = pd.to_numeric(prediction["true"], errors="raise").to_numpy()
            if not np.allclose(native_target, true, atol=2e-5, rtol=2e-5):
                difference = float(np.max(np.abs(native_target - true)))
                raise RuntimeError(
                    f"ALIGNN target values do not match the eV cache for split={split}; "
                    f"maximum absolute difference={difference:.6g}."
                )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            if split == "test":
                raise
            # Some official ALIGNN releases export only complete train/validation
            # batches and omit the remainder. Such partial outputs are not valid
            # for split metrics, but they must not invalidate the complete test run.
            print(
                f"WARNING: omitting incomplete {split} prediction export: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        prediction["row_index"] = prediction["xyz_file"].map(
            expected_by_name["row_index"]
        ).astype(int)
        prediction["sample_id"] = prediction["xyz_file"].map(
            expected_by_name["sample_id"]
        )
        prediction["N"] = prediction["xyz_file"].map(expected_by_name["N"]).astype(int)
        prediction["model"] = MODEL_NAME
        prediction["protocol"] = protocol
        prediction["seed"] = seed
        prediction["split"] = split
        prediction["unit"] = "eV"
        frames.append(
            prediction[
                [
                    "model",
                    "protocol",
                    "seed",
                    "split",
                    "row_index",
                    "sample_id",
                    "N",
                    "true",
                    "pred",
                    "unit",
                    "xyz_file",
                ]
            ]
        )
    result = pd.concat(frames, ignore_index=True)
    test_count = int(result["split"].eq("test").sum())
    if test_count != EXPECTED_COUNTS[protocol]["test"]:
        raise RuntimeError(
            f"Collected {test_count} test predictions; expected "
            f"{EXPECTED_COUNTS[protocol]['test']}."
        )
    if not np.isfinite(result[["true", "pred"]].to_numpy(dtype=float)).all():
        raise RuntimeError("ALIGNN predictions contain NaN or infinite values.")
    return result


def metric_values(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = pred - true
    ss_total = float(np.sum((true - np.mean(true)) ** 2))
    ss_residual = float(np.sum(error**2))
    r2 = float("nan") if ss_total <= 0 else 1.0 - ss_residual / ss_total
    return {
        "r2": r2,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias_pred_minus_true": float(np.mean(error)),
    }


def package_version() -> str:
    try:
        return importlib.metadata.version("alignn")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def write_training_history(native_dir: Path, run_dir: Path) -> tuple[int | None, float | None]:
    train_path = locate_output(native_dir, "history_train.json")
    val_path = locate_output(native_dir, "history_val.json")
    if train_path is None and val_path is None:
        return None, None

    def losses(path: Path | None) -> list[float]:
        if path is None:
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = []
        for row in payload if isinstance(payload, list) else []:
            value = row[0] if isinstance(row, list) and row else row
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                result.append(float("nan"))
        return result

    train = losses(train_path)
    val = losses(val_path)
    length = max(len(train), len(val))
    frame = pd.DataFrame({"epoch": np.arange(1, length + 1)})
    if train:
        frame["train_loss"] = pd.Series(train)
    if val:
        frame["validation_loss"] = pd.Series(val)
    atomic_csv(frame, run_dir / "training_history.csv")
    finite = np.asarray(val, dtype=float)
    if len(finite) and np.isfinite(finite).any():
        best_index = int(np.nanargmin(finite))
        return best_index + 1, float(finite[best_index])
    return None, None


def write_metrics(
    predictions: pd.DataFrame,
    run_dir: Path,
    protocol: str,
    seed: int,
    best_epoch: int | None,
    best_val_loss: float | None,
) -> pd.DataFrame:
    rows = []
    for split in SPLIT_ORDER:
        part = predictions[predictions["split"].eq(split)]
        if part.empty:
            continue
        values = metric_values(
            part["true"].to_numpy(dtype=float), part["pred"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "model": MODEL_NAME,
                "protocol": protocol,
                "seed": seed,
                "split": split,
                "count": len(part),
                "unit": "eV",
                **values,
                "best_epoch": best_epoch,
                "validation_selection_score": (
                    -best_val_loss if best_val_loss is not None else np.nan
                ),
                "alignn_version": package_version(),
            }
        )
    frame = pd.DataFrame(rows)
    atomic_csv(frame, run_dir / "metrics.csv")
    return frame


def completed_run_is_valid(run_dir: Path, protocol: str, seed: int) -> bool:
    metrics_path = run_dir / "metrics.csv"
    predictions_path = run_dir / "predictions.csv"
    status_path = run_dir / "run_status.txt"
    if not (metrics_path.is_file() and predictions_path.is_file() and status_path.is_file()):
        return False
    try:
        metrics = pd.read_csv(metrics_path)
        predictions = pd.read_csv(predictions_path)
    except Exception:
        return False
    test = metrics[metrics["split"].astype(str).str.lower().eq("test")]
    if len(test) != 1 or int(test.iloc[0]["count"]) != EXPECTED_COUNTS[protocol]["test"]:
        return False
    if str(test.iloc[0].get("unit", "")).lower() != "ev":
        return False
    test_predictions = predictions[
        predictions["split"].astype(str).str.lower().eq("test")
    ]
    if len(test_predictions) != EXPECTED_COUNTS[protocol]["test"]:
        return False
    return (
        set(predictions["protocol"].astype(str)) == {protocol}
        and set(pd.to_numeric(predictions["seed"], errors="coerce").dropna().astype(int))
        == {seed}
        and "COMPLETED" in status_path.read_text(encoding="utf-8", errors="replace")
    )


def validate_train_command(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("--train-command is empty.")
    executable = parts[0]
    if Path(executable).is_file():
        parts[0] = str(Path(executable).resolve())
    elif shutil.which(executable) is None:
        raise FileNotFoundError(
            f"ALIGNN training command was not found: {executable}. Activate the "
            "my_alignn environment and verify 'which train_alignn.py'."
        )
    return parts


def safely_remove_native(native_dir: Path, run_dir: Path) -> None:
    if not native_dir.exists():
        return
    resolved_native = native_dir.resolve()
    resolved_run = run_dir.resolve()
    if resolved_native.parent != resolved_run:
        raise RuntimeError(f"Refusing to remove output outside run directory: {native_dir}")
    shutil.rmtree(native_dir)


def run_training(
    command_prefix: list[str],
    input_dir: Path,
    run_dir: Path,
    native_dir: Path,
    ordered: pd.DataFrame,
    protocol: str,
    seed: int,
    resume: bool,
) -> pd.DataFrame | None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume and completed_run_is_valid(run_dir, protocol, seed):
        print(f"SKIP completed ALIGNN run: protocol={protocol} seed={seed}", flush=True)
        return pd.read_csv(run_dir / "metrics.csv")

    safely_remove_native(native_dir, run_dir)
    native_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "run_status.txt"
    status_path.write_text(
        f"RUNNING\nmodel=alignn\nprotocol={protocol}\nseed={seed}\nunit=eV\n",
        encoding="utf-8",
    )
    command = command_prefix + [
        "--root_dir",
        str(input_dir),
        "--config_name",
        str(input_dir / "config.json"),
        "--file_format",
        "xyz",
        "--output_dir",
        str(native_dir),
    ]
    print("COMMAND: " + " ".join(shlex.quote(part) for part in command), flush=True)
    log_path = run_dir / "train_stdout.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=input_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        status_path.write_text(
            f"FAILED\nmodel=alignn\nprotocol={protocol}\nseed={seed}\n"
            f"return_code={return_code}\nlog={log_path}\n",
            encoding="utf-8",
        )
        raise subprocess.CalledProcessError(return_code, command)

    predictions = validate_and_collect_predictions(native_dir, ordered, protocol, seed)
    atomic_csv(predictions, run_dir / "predictions.csv")
    best_epoch, best_val_loss = write_training_history(native_dir, run_dir)
    metrics = write_metrics(
        predictions, run_dir, protocol, seed, best_epoch, best_val_loss
    )
    test = metrics[metrics["split"].eq("test")].iloc[0]
    status_path.write_text(
        "COMPLETED\n"
        f"model=alignn\nprotocol={protocol}\nseed={seed}\nunit=eV\n"
        f"test_count={int(test['count'])}\n"
        f"test_r2={float(test['r2']):.12g}\n"
        f"test_mae={float(test['mae']):.12g}\n"
        f"test_rmse={float(test['rmse']):.12g}\n",
        encoding="utf-8",
    )
    print(
        f"DONE ALIGNN {protocol} seed={seed}: R2={test['r2']:.6f} "
        f"MAE={test['mae']:.6f} eV RMSE={test['rmse']:.6f} eV",
        flush=True,
    )
    return metrics


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_names = ("r2", "mae", "rmse", "bias_pred_minus_true")
    for (model, protocol, split), part in frame.groupby(
        ["model", "protocol", "split"], sort=True
    ):
        row: dict[str, Any] = {
            "model": model,
            "protocol": protocol,
            "split": split,
            "n_seeds": int(part["seed"].nunique()),
            "unit": "eV",
        }
        for name in metric_names:
            values = pd.to_numeric(part[name], errors="coerce").dropna()
            mean = float(values.mean()) if len(values) else float("nan")
            std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
            row[f"{name}_mean_pm_std"] = (
                f"{mean:.6f} +/- {std:.6f}" if math.isfinite(std) else f"{mean:.6f}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metrics(benchmark_dir: Path) -> None:
    paths = sorted((benchmark_dir / "runs").glob("*/*/seed_*/metrics.csv"))
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
            continue
        required = {
            "model",
            "protocol",
            "seed",
            "split",
            "count",
            "unit",
            "r2",
            "mae",
            "rmse",
            "bias_pred_minus_true",
        }
        if required.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        print("No completed benchmark metric files were found.", flush=True)
        return
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(
        subset=["model", "protocol", "seed", "split"], keep="last"
    ).sort_values(["protocol", "model", "seed", "split"])
    atomic_csv(combined, benchmark_dir / "per_seed_metrics.csv")
    summary = summary_table(combined)
    atomic_csv(summary, benchmark_dir / "metrics_mean_std.csv")
    atomic_csv(
        summary[summary["split"].eq("test")],
        benchmark_dir / "test_metrics_mean_std.csv",
    )
    alignn_rows = combined[combined["model"].astype(str).str.lower().eq(MODEL_NAME)]
    atomic_csv(alignn_rows, benchmark_dir / "alignn_per_seed_metrics.csv")

    completed = {
        (str(row.protocol), int(row.seed))
        for row in alignn_rows[alignn_rows["split"].eq("test")].itertuples()
    }
    expected = {(protocol, seed) for protocol in EXPECTED_COUNTS for seed in EXPECTED_SEEDS}
    missing = sorted(expected - completed)
    atomic_json(
        benchmark_dir / "alignn_completion_status.json",
        {
            "complete": not missing,
            "completed_runs": [
                {"protocol": protocol, "seed": seed} for protocol, seed in sorted(completed)
            ],
            "missing_runs": [
                {"protocol": protocol, "seed": seed} for protocol, seed in missing
            ],
            "expected_run_count": 10,
            "completed_run_count": len(completed),
        },
    )
    print(
        f"Aggregated {len(completed)}/10 ALIGNN runs; missing={len(missing)}.",
        flush=True,
    )


def selected_tasks(args: argparse.Namespace) -> list[tuple[str, int]]:
    invalid_seeds = sorted(set(args.seeds) - set(EXPECTED_SEEDS))
    if invalid_seeds:
        raise ValueError(
            f"Only the registered five seeds are allowed: {EXPECTED_SEEDS}; "
            f"received {invalid_seeds}."
        )
    tasks = [
        (protocol, seed) for protocol in args.protocols for seed in args.seeds
    ]
    if args.task_index is not None:
        canonical = [
            (protocol, seed)
            for protocol in ("random", "group")
            for seed in EXPECTED_SEEDS
        ]
        if args.task_index < 0 or args.task_index >= len(canonical):
            raise ValueError("--task-index must be between 0 and 9.")
        tasks = [canonical[args.task_index]]
    return tasks


def main() -> None:
    args = parse_args()
    project_dir, benchmark_dir, cache_path, work_dir = resolve_paths(args)
    print(f"Project: {project_dir}", flush=True)
    print(f"Benchmark: {benchmark_dir}", flush=True)
    print(f"ALIGNN work directory: {work_dir}", flush=True)
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"Benchmark directory was not found: {benchmark_dir}")
    if args.aggregate_only:
        aggregate_metrics(benchmark_dir)
        return

    tasks = selected_tasks(args)
    shared = prepare_shared_dataset(
        benchmark_dir, cache_path, work_dir, args.force_data
    )
    prepared: list[tuple[str, int, Path, Path, Path, pd.DataFrame]] = []
    for protocol, seed in tasks:
        input_dir, run_dir, native_dir, ordered = prepare_run_input(
            protocol,
            seed,
            benchmark_dir,
            work_dir,
            shared,
            args.epochs,
            args.num_workers,
        )
        prepared.append((protocol, seed, input_dir, run_dir, native_dir, ordered))
        print(
            f"Prepared {protocol} seed={seed}: "
            f"{EXPECTED_COUNTS[protocol]} -> {input_dir}",
            flush=True,
        )
    if args.prepare_only:
        print("Preparation and validation completed; training was not started.", flush=True)
        return

    command_prefix = validate_train_command(args.train_command)
    for protocol, seed, input_dir, run_dir, native_dir, ordered in prepared:
        run_training(
            command_prefix,
            input_dir,
            run_dir,
            native_dir,
            ordered,
            protocol,
            seed,
            args.resume,
        )
        aggregate_metrics(benchmark_dir)
    aggregate_metrics(benchmark_dir)


if __name__ == "__main__":
    main()
