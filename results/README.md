# Generated results

Training outputs are written under `results/random/seed_<seed>/` and
`results/group/seed_<seed>/`. Each run contains `mode_random/` or
`mode_group_folder/` with the original model's results and diagnostics.

`results/processed_data.pt` is a generated preprocessing cache. Run
`python -m scripts.evaluate` to create `results/metrics_summary.csv`.

Generated caches, model weights, and experiment outputs are ignored by Git.
This folder intentionally contains no precomputed performance claim.
