# Fullerene HOMO–LUMO GNN

Data and training-code package for the proposed physics-guided fullerene
HOMO–LUMO gap model accompanying *Size–Structure Decomposition of Fullerene
Gaps with Physics-Guided Residual Graph Learning*. It includes 7,035 labeled
neutral fullerene structures. Benchmark, ablation, and analysis entry points
are provided, but the full manuscript results have not been regenerated from
this package.

## Contents

- `data/dataset.csv`: 7,035 labels and metadata records.
- `data/structure_manifest.csv`: sample identifiers and relative geometry paths.
- `data/structures/C*/`: corresponding Gaussian `.gjf` geometries by cage size.
- `data/LICENSE.md`: CC BY 4.0 terms for the dataset and split manifests.
- `splits/random/`: fixed 8:1:1 splits for seeds 2024–2028.
- `splits/group/group_split.csv`: fixed cage-size-disjoint group split for cross-size transfer.
- `src/`: proposed model, features, validated loader, and preserved training core.
- `scripts/`: preprocessing, training, evaluation, graph/ALIGNN benchmarks,
  ablations, and physical/error analyses.
- `configs/config.yaml`: main-model settings and seeds.
- `results/README.md`: generated output layout; results are not committed.
- `checksums/SHA256SUMS.txt`: packaged-file integrity checksums.
- `LICENSE` and `THIRD_PARTY_NOTICES.md`: code license and upstream notices.

## Dataset and calculations

- C20–C60: the supplied topologically distinct isomer collection, with matched
  labels and geometries for every **included** sample. The labeled table is
  not a complete all-isomer enumeration: C54/ID 43 has a source geometry but
  no matching label and is excluded (see `data/README.md`).
- C70–C100: IPR isomers only, as represented in the supplied data.
- C62–C68: absent; no classical isolated-pentagon-rule fullerene cage exists
  in this interval ([mathematical discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC2614729/)).
- Gaussian input route: `B3LYP/6-31G* opt pop=NPA`; the basis set is also
  denoted `B3LYP/6-31G(d)` in the manuscript (`*` and `(d)` denote the same
  polarization convention here). This package supplies input geometries and
  labels, not Gaussian output logs or a DFT recalculation workflow.
- `HOMO`, `LUMO`, and `Gap` are in Hartree in the CSV. Training converts `Gap`
  to eV using 27.211386245988 eV/Hartree.

## Split definitions

- Random: 5,628 train, 703 validation, and 704 test records per seed.
- Group: C20–C56 train (2,752), C58/C60 validation (3,017), and C70–C100
  test (1,266). The group test differs in both cage size and isomer
  composition; it is not a pure size-extrapolation experiment.

Manifest paths are relative to this repository. Local source paths, model
checkpoints, caches, generated figures, Gaussian `.log` files, and derived
ALIGNN `.xyz` files are excluded.

## Proposed-model quick start

From the repository root, use Python 3.12 and install a PyTorch build suitable
for your CPU/CUDA setup, then:

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

Repeat random training for seeds 2025–2028 as needed. Preprocessing creates
`results/processed_data.pt`; the wrapper checks its split against the fixed
manifest. The original proposed-model algorithm is retained in
`src/legacy_gap2.py`; `src/model.py` and `src/features.py` expose its APIs.

## Reproducing the paper results

These are **entry points and expected artifacts**, not a claim that the
publication tables/figures have been validated end-to-end in this repository.
Run commands from the repository root. For graph benchmarks/ablations install
`requirements-benchmarks.txt` (including PyTorch Geometric). Official ALIGNN
requires its separate upstream installation and compatible runtime; its
`train_alignn.py` CLI is not bundled. All-model five-seed runs can require
substantial GPU time.

| Manuscript item | Command(s) | Expected output / limitation |
| --- | --- | --- |
| Table 1, proposed vs. Matformer and FullereneNet | `python -m scripts.benchmarks.run_graph_benchmarks` | `results/graph_benchmarks/test_metrics_mean_std.csv`; check `incomplete_runs.json`. This is a separate benchmark implementation from the quick-start trainer. |
| Table 1, ALIGNN | `python -m scripts.benchmarks.run_alignn_benchmark --train-command train_alignn.py` | Adds ALIGNN metrics to the graph-benchmark summary; first run graph benchmark for the validated cache/manifests. Requires official ALIGNN. |
| Table 2, module/input ablations | `python -m scripts.ablations.run_module_input_ablation` | `results/ablations/test_metrics_mean_std.csv`; compare exact configurations/protocols with the final manuscript. |
| Figures 3–4, model comparisons and predictions | `python -m scripts.analysis.draw_paper_results` | `results/paper_figures/raw/comparisons/`, `proposed/`, `tables/`; requires completed runs; inspect `plot_manifest.json`. |
| Figure 5, fixed-size correlations | `python -m scripts.analysis.plot_physical_gap_analysis` | `results/physical_analysis/`, including `fixed_N_spearman_correlations.csv`; extracts descriptors from packaged geometries as needed. |
| Figures 6–7, error/size analyses | `python -m scripts.analysis.draw_paper_results` | `results/paper_figures/raw/` error and cage-size outputs; check filenames against final panel captions. |

The figure-number associations reflect the proposed manuscript outline; exact
panel-to-file mapping and numerical agreement must be checked against the
final manuscript and completed runs. The plotting script records missing
result sources instead of fabricating figures. See `results/README.md`.

## Citation and contact

If you use this dataset or code, please cite the accompanying manuscript:
*Size–Structure Decomposition of Fullerene Gaps with Physics-Guided Residual
Graph Learning*. Update this citation with the publication DOI when available.
For questions, open a [GitHub Issue](https://github.com/lizelong9361-art/Fullerene-HOMO-LUMO-GNN/issues).

## Licensing and release status

Original project code is available under the [MIT License](LICENSE).
The 7,035-row dataset, matched geometries, and predefined CSV split manifests
are available under [CC BY 4.0](data/LICENSE.md); please credit the repository
and accompanying manuscript and indicate modifications. Embedded third-party
model code retains its original MIT copyright and permission notices; see
[third-party notices](THIRD_PARTY_NOTICES.md). Dependency licenses remain with
their respective packages.

No `v1.0.0` release or Zenodo DOI is claimed here. Once the result mapping and
repository snapshot are confirmed, tag that snapshot and archive its GitHub
release with Zenodo. Cite the resulting version-specific DOI in the paper.
