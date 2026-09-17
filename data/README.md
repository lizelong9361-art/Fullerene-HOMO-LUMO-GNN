# Data layout

`dataset.csv` is the canonical 7,035-row metadata and label table.
`structure_manifest.csv` maps its zero-based `row_index` to a stable `sample_key`
and one relative `.gjf` path.

Structure filenames use this convention:

```text
C{AtomCount}_{ID padded to 6 digits}.gjf
```

Example: metadata row `AtomCount=60, ID=1` maps to
`data/structures/C60/C60_000001.gjf`.

The source geometry collection contained one extra file at C54/ID 43 with no
matching metadata row. That orphan file is deliberately excluded, leaving a
one-to-one set of 7,035 metadata rows and 7,035 geometry files.

The dataset and matched geometry files are licensed under
[CC BY 4.0](LICENSE.md). The predefined CSV split manifests in `../splits/`
have the same license. Original project code has a separate MIT license.
