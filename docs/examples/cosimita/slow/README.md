# Long-running cosimita examples

Notebooks here are **not executed by CI**. The `notebooks` job in
`.github/workflows/ci.yml` globs `docs/examples/cosimita/*.ipynb`, which does
not recurse, so anything in this directory is skipped by construction.

They live here because each one runs real simulations to a few hundred
triggered events per panel, which takes minutes rather than seconds. The
numbered notebooks in the parent directory are deliberately kept fast enough
to gate every push; these are for reading, and for re-running by hand when
the physics they illustrate changes.

Re-run one with:

    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1200 <notebook>

Do **not** pass `MPLBACKEND=Agg` when re-executing: it overrides the inline
backend and silently strips every figure from the committed `.ipynb` while
still exiting 0.

| notebook | runtime | what it shows |
|---|---|---|
| `source_placement_gallery.ipynb` | ~150 s | Measured energy and Compton Data Space for six `NearPointSource` placements and three `ExtendedSource` widths. |
