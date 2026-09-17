"""Train the original physics-guided model with the published fixed splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import legacy_gap2 as core
from .dataset import load_or_build, manifest_records, repository_root


def expected_splits(n_samples: int, atom_counts: np.ndarray,
                    protocol: str, seed: int) -> np.ndarray:
    labels = np.empty(n_samples, dtype=object)
    if protocol == "random":
        permutation = np.random.RandomState(seed).permutation(n_samples)
        n_train, n_val = int(0.8 * n_samples), int(0.1 * n_samples)
        labels[permutation[:n_train]] = "train"
        labels[permutation[n_train:n_train + n_val]] = "val"
        labels[permutation[n_train + n_val:]] = "test"
    elif protocol == "group":
        for i, n in enumerate(atom_counts):
            if 20 <= n <= 56:
                labels[i] = "train"
            elif n in (58, 60):
                labels[i] = "val"
            elif 70 <= n <= 100:
                labels[i] = "test"
            else:
                raise ValueError(f"C{n} is outside the declared group split")
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    return labels


def validate_fixed_split(root: Path, protocol: str, seed: int) -> dict[str, int]:
    frame, _ = manifest_records(root)
    manifest_path = (root / "splits" / "random" / f"random_seed{seed}.csv"
                     if protocol == "random" else root / "splits" / "group" / "group_split.csv")
    fixed = pd.read_csv(manifest_path)
    if len(fixed) != len(frame) or not np.array_equal(
            fixed["row_index"].to_numpy(dtype=int), np.arange(len(frame))):
        raise ValueError(f"Incomplete or out-of-order fixed split: {manifest_path}")
    keys = [f"C{int(n)}_{int(sample_id):06d}"
            for n, sample_id in zip(frame["AtomCount"], frame["ID"])]
    if fixed["sample_key"].tolist() != keys:
        raise ValueError(f"Sample-key mismatch in {manifest_path}")
    expected = expected_splits(len(frame), frame["AtomCount"].to_numpy(dtype=int),
                               protocol, seed)
    if not np.array_equal(fixed["split"].to_numpy(), expected):
        raise ValueError(f"Core split does not match fixed manifest: {manifest_path}")
    return fixed["split"].value_counts().to_dict()


def main(protocol_default: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("random", "group"),
                        default=protocol_default or "group")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--config", type=Path, default=repository_root() / "configs/config.yaml")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the data and fixed split without training")
    args = parser.parse_args()
    if protocol_default and args.protocol != protocol_default:
        parser.error(f"This entry point only accepts --protocol {protocol_default}")

    root = repository_root()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if args.seed not in config.get("seeds", [2024, 2025, 2026, 2027, 2028]):
        parser.error("Seed is not listed in config.yaml")
    counts = validate_fixed_split(root, args.protocol, args.seed)
    print(f"Validated {args.protocol} seed {args.seed}: {counts}")
    if args.dry_run:
        return

    for config_key, core_name in (("batch_size", "BATCH_SIZE"),
                                  ("learning_rate", "LEARNING_RATE"),
                                  ("max_epochs", "MAX_EPOCHS"),
                                  ("patience", "PATIENCE"),
                                  ("weight_decay", "WEIGHT_DECAY")):
        if config_key in config:
            setattr(core, core_name, config[config_key])
    core.SEED = args.seed
    core.ENABLE_POSTPROCESSING = False
    run_dir = root / "results" / args.protocol / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    core.BASE_SAVE_DIR = str(run_dir)
    core.seed_everything(args.seed)
    data = load_or_build(root, root / "results" / "processed_data.pt",
                         rebuild=args.rebuild_cache,
                         workers=int(config.get("preprocess_workers", 4)))
    groups = [core.split_for_atom_count(int(item["intended_N"])) for item in data]
    mode = "random" if args.protocol == "random" else "group_folder"
    output, r2 = core.run_training_experiment(data, groups, split_mode=mode)
    print(f"Results: {output}; test R2: {r2:.6f}")


if __name__ == "__main__":
    main()
