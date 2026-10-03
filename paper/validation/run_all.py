"""
Run every validation of the paper and write ``paper/results/validation.json``.

- ``validate_kwant.py`` and ``validate_pythtb.py``: cross-checks against the
  pinned Kwant and PythTB (see ``paper/environment.yml``).
- The gallery examples of ``analytic_catalogue.ANALYTIC``, re-run here: each
  asserts its closed-form result.

Usage (from the repository root)::

    python paper/validation/run_all.py
"""
import contextlib
import importlib.metadata as metadata
import io
import json
import platform
import runpy
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from analytic_catalogue import ANALYTIC  # noqa: E402


def run(path):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        namespace = runpy.run_path(str(path), run_name='__main__')
    plt.close('all')
    return namespace, out.getvalue()


def main():
    results = {'versions': {p: metadata.version(p) for p in
                            ('tbkit', 'kwant', 'pythtb', 'numpy', 'scipy')},
               'platform': platform.platform(), 'python': platform.python_version(),
               'cross_checks': {}, 'analytic': []}
    for name in ('validate_kwant.py', 'validate_pythtb.py'):
        namespace, out = run(HERE / name)
        print(out, end='')
        results['cross_checks'][name[len('validate_'):-3]] = namespace['RESULTS']
    for entry in ANALYTIC:
        start = time.perf_counter()
        run(ROOT / 'examples' / entry['example'])
        seconds = time.perf_counter() - start
        results['analytic'].append({**entry, 'passed': True, 'seconds': round(seconds, 2)})
        print(f"passed  {entry['example']:55s} {seconds:6.1f} s")
    out = ROOT / 'paper' / 'results' / 'validation.json'
    out.write_text(json.dumps(results, indent=2) + '\n')
    print(f'Wrote {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
