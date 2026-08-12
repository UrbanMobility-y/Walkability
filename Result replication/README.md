# Result notebooks

`Figure 1.ipynb` through `Figure 4.ipynb` contain two complementary sections:

1. **Portable implementation.** These cells read the public source data and
   write regenerated outputs to `outputs/figures`.
2. **Authors' executed workstation record.** These cells retain the exact code,
   absolute paths, execution counts and outputs from the authors' validation
   run. GitHub displays each output directly beneath the code that generated it,
   allowing readers to inspect the provenance of the corresponding paper
   results without executing the notebooks.

Use the first section to run the analysis on another computer. The absolute
paths in the second section document the original execution environment and are
not required by the portable section.

Download the data from
[Zenodo (DOI: 10.5281/zenodo.21902140)](https://zenodo.org/records/21902140)
and use this layout:

```text
Code and Data/
├── Github code/
└── source data/
```

Run `jupyter notebook` from the repository root. If the data are stored
elsewhere, set the `WALKABILITY_SOURCE_DATA` environment variable to the full
path before starting Jupyter.

The released fields support the analytical plots but not all production
cartography or the five-minute exposure calculations. In particular, the
portable Figure 1 city panel uses released city coordinates because the
production boundary layers are not in the Zenodo archive; the authors'
workstation record retains the paper's original panel. See `REPRODUCIBILITY.md`
for the full boundary.
