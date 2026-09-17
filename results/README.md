# Generated results (not versioned)

- `random/seed_<seed>/`, `group/seed_<seed>/`: proposed-model training runs.
- `processed_data.pt`: proposed-model preprocessing cache.
- `metrics_summary.csv`: summary from `python -m scripts.evaluate`.
- `graph_benchmarks/`: graph/ALIGNN manifests, validated cache, per-seed
  metrics, `test_metrics_mean_std.csv`, and `incomplete_runs.json` if needed.
- `alignn/`: derived XYZ inputs and ALIGNN workspace.
- `ablations/`: module/input-ablation runs and `test_metrics_mean_std.csv`.
- `physical_analysis/`: descriptors, fixed-size correlations, figures.
- `paper_figures/raw/`: available comparisons, proposed-model predictions/error figures,
  tables, and `plot_manifest.json` documenting missing inputs.

Results, checkpoints, and generated caches are ignored by Git. No
precomputed performance claim or publication figure is bundled.
