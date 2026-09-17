# Reproducible splits

Each split manifest contains `row_index`, `sample_key`, `ID`, `AtomCount`,
`split`, and the relative `structure_file` path.

- `random/random_seed2024.csv` through `random_seed2028.csv` preserve the five
  fixed random 8:1:1 experiments.
- `group/group_split.csv` preserves the size-extrapolation experiment:
  C20-C56 train, C58/C60 validation, and C70-C100 test.

The five original group-seed manifests had identical assignments, so they are
represented by one non-duplicated group manifest.
