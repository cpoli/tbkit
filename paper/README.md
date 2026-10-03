# The tbkit paper

Everything behind the manuscript, versioned with the code it describes.

| Directory | Contents |
|---|---|
| `validation/` | Cross-checks against Kwant and PythTB, and the catalogue of closed-form results asserted by the gallery |
| `benchmarks/` | Time and memory against Kwant and PythTB |
| `figures/` | One script per figure of the manuscript |
| `results/` | JSON written by the scripts above, read by the figures and tables |
| `manuscript/` | LaTeX source (SciPost template) |

## Reproducing

```bash
micromamba create -f paper/environment.yml && micromamba activate tbkit-paper
pip install -e .
python paper/validation/run_all.py        # asserts every cross-check (about 1 min)
python paper/benchmarks/run_benchmarks.py # about 15 min, one thread
python paper/figures/build_all.py         # figures and tables from results/
cd paper/manuscript && latexmk -pdf tbkit.tex
```

The validation asserts every comparison, so a failing check stops the run.
Benchmark timings depend on the machine. `results/benchmarks.json` records
the platform it ran on.
