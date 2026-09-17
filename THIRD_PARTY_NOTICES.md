# Third-party notices

The graph benchmark in `scripts/benchmarks/run_graph_benchmarks.py` embeds
adapted model implementations for comparisons. These portions are not claimed
as original project code. Preserve the upstream copyright and permission
notices when redistributing them:

- FullereneNet: [Liu's Group / FullereneNet](https://github.com/Liu-Group-UF/FullereneNet), MIT; see
  [`third_party_licenses/FullereneNet-LICENSE`](third_party_licenses/FullereneNet-LICENSE).
- Matformer: [YKQ98 / Matformer](https://github.com/YKQ98/Matformer), MIT; see
  [`third_party_licenses/Matformer-LICENSE`](third_party_licenses/Matformer-LICENSE).

The original project code is licensed under the repository-root `LICENSE`.
The dataset and split manifests are licensed separately under `data/LICENSE.md`.
External packages installed through `requirements.txt` and
`requirements-benchmarks.txt` retain their own licenses and are not bundled
into this repository. Official ALIGNN is not bundled.
