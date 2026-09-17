# Fullerene HOMO-LUMO GNN

Reproducible data and training-code package for a physics-guided fullerene
HOMO-LUMO gap model. It contains 7,035 neutral fullerene samples and their
Gaussian input geometries.

## Contents

- `data/dataset.csv`: labels and descriptors for all 7,035 samples.
- `data/structure_manifest.csv`: stable sample identifiers and relative geometry paths.
- `data/structures/C*/`: Gaussian `.gjf` files grouped by carbon atom count.
- `splits/random/`: fixed 8:1:1 train/validation/test splits for seeds 2024-2028.
- `splits/group/group_split.csv`: fixed size-based extrapolation split.
- `src/`: validated data loader, model and feature APIs, original audited
  training core, and training orchestration.
- `scripts/`: preprocessing, random/group training, and evaluation entry points.
- `configs/config.yaml`: experiment settings and random seeds.
- `results/README.md`: output layout; generated results/checkpoints are not committed.
- `checksums/SHA256SUMS.txt`: SHA-256 checksums for integrity verification.

## Dataset summary

- Carbon sizes: C20-C60 and C70-C100 (even sizes represented in the source data).
- Geometry calculation route found in the supplied inputs: `B3LYP/6-31G* opt pop=NPA`.
- The `HOMO`, `LUMO`, and `Gap` columns are stored in Hartree in the source CSV.
- Training converts `Gap` to eV using 27.211386245988 eV/Hartree.

## Split definitions

- Random: 5,628 train, 703 validation, and 704 test samples for each seed.
- Group: C20-C56 train (2,752), C58/C60 validation (3,017), and C70-C100 test (1,266).

All manifest paths are relative to this package. Local Windows/Linux source paths,
model checkpoints, caches, result figures, Gaussian `.log` outputs, and derived
ALIGNN `.xyz` files are intentionally excluded.

## Run from the repository root

Use Python 3.12 with the packages in `requirements.txt`. Install the appropriate
PyTorch build for your CPU/CUDA environment, then install the remaining packages.

```bash
python -m pip install -r requirements.txt
python -m scripts.preprocess --validate-only
python -m scripts.train_random_split --seed 2024 --dry-run
python -m scripts.train_group_split --seed 2024 --dry-run
python -m scripts.preprocess
python -m scripts.train_random_split --seed 2024
python -m scripts.train_group_split --seed 2024
python -m scripts.evaluate
```

Repeat the random training command with seeds 2025-2028 as needed. Preprocessing
creates `results/processed_data.pt` from the CSV/GJF files. Training checks
that the original algorithm's split exactly matches the published fixed
manifest before using it. The physics-guided model and feature calculations
remain in `src/legacy_gap2.py`, copied from the supplied `gap2.py` to preserve
the training behavior; `src/model.py` and `src/features.py` expose its APIs.

The repository packages the main physics-guided model. ALIGNN, Matformer,
FullereneNet, ablations, plotting scripts, and their historical outputs are not
part of this minimal run path. No expensive training run was performed while
preparing this upload package.

## Notes before publication

`LICENSE` is a placeholder pending the copyright holder's choice. Replace it
with an actual permission grant before public release. The dataset provenance
and methods should also be described in the final paper.
