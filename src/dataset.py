"""Validate the portable CSV/GJF mapping and build the original graph features."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pandas as pd
import torch

from . import legacy_gap2 as core


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def manifest_records(root: Path | None = None) -> tuple[pd.DataFrame, list[tuple]]:
    root = (root or repository_root()).resolve()
    csv_path = root / "data" / "dataset.csv"
    manifest_path = root / "data" / "structure_manifest.csv"
    frame = pd.read_csv(csv_path)
    manifest = pd.read_csv(manifest_path)
    required = {"ID", "AtomCount", "Gap"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing dataset columns: {sorted(required - set(frame.columns))}")
    if len(frame) != 7035 or len(manifest) != len(frame):
        raise ValueError("Expected 7,035 matching metadata and manifest rows")
    if frame[["ID", "AtomCount", "Gap"]].isna().any().any():
        raise ValueError("ID, AtomCount, and Gap must be non-null")
    if frame.duplicated(["AtomCount", "ID"]).any():
        raise ValueError("Duplicate (AtomCount, ID) in dataset")
    if manifest["sample_key"].duplicated().any():
        raise ValueError("Duplicate sample_key in structure manifest")

    tasks = []
    for index, (row, entry) in enumerate(zip(frame.itertuples(index=False), manifest.itertuples(index=False))):
        n, sample_id = int(row.AtomCount), int(row.ID)
        expected_key = f"C{n}_{sample_id:06d}"
        if (int(entry.row_index) != index or entry.sample_key != expected_key
                or int(entry.AtomCount) != n or int(entry.ID) != sample_id):
            raise ValueError(f"Metadata/manifest mismatch at row {index}")
        relative = Path(entry.structure_file)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe structure path: {relative}")
        expected_path = Path("data") / "structures" / f"C{n}" / f"{expected_key}.gjf"
        if relative.as_posix() != expected_path.as_posix():
            raise ValueError(f"Unexpected structure path at row {index}: {relative}")
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        tasks.append((str(source), float(row.Gap) * core.HARTREE_TO_EV, n, sample_id))
    return frame, tasks


def source_fingerprint(root: Path | None = None) -> str:
    root = root or repository_root()
    digest = sha256()
    for relative in (Path("data/dataset.csv"), Path("data/structure_manifest.csv")):
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def load_or_build(root: Path | None = None, cache_path: Path | None = None,
                  rebuild: bool = False, workers: int = 4) -> list[dict]:
    root = (root or repository_root()).resolve()
    cache_path = cache_path or root / "results" / "processed_data.pt"
    _, tasks = manifest_records(root)
    fingerprint = source_fingerprint(root)
    if cache_path.is_file() and not rebuild:
        cache = core.safe_torch_load(str(cache_path), map_location="cpu")
        if (isinstance(cache, dict)
                and cache.get("schema_version") == core.CACHE_SCHEMA_VERSION
                and cache.get("source_fingerprint") == fingerprint
                and cache.get("target_unit") == core.TARGET_UNIT
                and len(cache.get("data", [])) == len(tasks)):
            return cache["data"]

    if workers < 1:
        raise ValueError("workers must be positive")
    print(f"Processing {len(tasks)} validated CSV/GJF pairs...")
    if workers == 1:
        processed = [core.process_single_file(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            processed = list(pool.map(core.process_single_file, tasks))
    errors = [item for item in processed if item is None or "error" in item]
    if errors:
        raise RuntimeError(f"Graph extraction failed for {len(errors)} samples; first error: {errors[0]}")
    for index, item in enumerate(processed):
        _, _, n, sample_id = tasks[index]
        if (int(item["N"]) != n or int(item["intended_N"]) != n
                or int(item["sample_id"]) != sample_id or item["unit"] != "eV"):
            raise RuntimeError(f"Processed graph mismatch at row {index}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": core.CACHE_SCHEMA_VERSION,
                "source_fingerprint": fingerprint, "target_unit": core.TARGET_UNIT,
                "data": processed}, cache_path)
    return processed
