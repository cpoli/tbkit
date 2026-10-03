"""
Build the figures and tables of the manuscript.

- fig_models, fig_topology, fig_response, fig_driven: three panels each,
  drawn by the panel functions of ``docs/make_readme_category_figures.py``
  (condensed gallery examples), relabelled (a)-(c) for print.
- fig_validation: the largest deviation of every cross-check.
- fig_benchmarks: time against problem size.
- tables/*.tex: the cross-checks and the analytic catalogue.

Reads ``paper/results/*.json`` (written by validation/run_all.py and
benchmarks/run_benchmarks.py). Writes to ``paper/manuscript/figures`` and
``paper/manuscript/tables``.

Usage (from the repository root)::

    python paper/figures/build_all.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'paper' / 'results'
FIGS = ROOT / 'paper' / 'manuscript' / 'figures'
TABLES = ROOT / 'paper' / 'manuscript' / 'tables'
sys.path.insert(0, str(ROOT / 'docs'))

# Categorical slots 1-3 of the reference palette, in fixed order; every
# series also has its own marker and line style, so color is never the only cue.
COLORS = {'tbkit': '#2a78d6', 'pythtb': '#eb6834', 'kwant': '#1baf7a'}
MARKERS = {'tbkit': 'o', 'pythtb': 's', 'kwant': '^'}
STYLES = {'tbkit': '-', 'pythtb': '--', 'kwant': ':'}
LABELS = {'tbkit': 'tbkit', 'pythtb': 'PythTB', 'kwant': 'Kwant'}
WIDTH = 7.0  # inches, full text width

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555555', 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
                     'pdf.fonttype': 42, 'savefig.bbox': 'tight'})


def panels(name):
    import make_readme_category_figures as readme
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3))
    getattr(readme, name)(axes)
    for letter, ax in zip('abc', axes):
        ax.set_title('')
        if ax.get_legend() is not None:  # the README sizes legends for the web
            ax.legend(fontsize=6, frameon=False, loc='best')
        ax.text(-0.02, 1.04, f'({letter})', transform=ax.transAxes, fontweight='bold', ha='right')
    fig.tight_layout()
    fig.savefig(FIGS / f'fig_{name}.pdf')
    plt.close(fig)


def validation():
    data = json.loads((RESULTS / 'validation.json').read_text())
    rows = [(ref, name, dev) for ref, checks in data['cross_checks'].items() for name, dev in checks.items()]
    fig, ax = plt.subplots(figsize=(0.75 * WIDTH, 0.2 * len(rows) + 0.6))
    for y, (ref, name, dev) in enumerate(rows):
        ax.hlines(y, 1e-17, max(dev, 1e-17), color='#cccccc', lw=1)
        ax.plot(max(dev, 1e-17), y, MARKERS[ref], color=COLORS[ref], ms=5)
    ax.set_yticks(range(len(rows)), [name for _, name, _ in rows])
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlim(1e-17, 1e-5)
    ax.set_xlabel('largest deviation from the reference package')
    for ref in ('kwant', 'pythtb'):
        ax.plot([], [], MARKERS[ref], color=COLORS[ref], label=f'vs {LABELS[ref]}')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(axis='x')
    fig.savefig(FIGS / 'fig_validation.pdf')
    plt.close(fig)


def benchmarks():
    data = json.loads((RESULTS / 'benchmarks.json').read_text())
    tasks = [('bands', 'band energies'), ('chern', 'Chern number'), ('wilson', 'Wilson loops')]
    norb = max(r['norb'] for r in data['kspace'])
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.1))
    for ax, (task, title) in zip(axes, tasks):
        for package in ('tbkit', 'pythtb'):
            rows = sorted((r for r in data['kspace'] if r['task'] == task and r['norb'] == norb
                           and r['package'] == package), key=lambda r: r['nk'])
            ax.loglog([r['nk'] ** 2 for r in rows], [r['seconds'] for r in rows], STYLES[package],
                      marker=MARKERS[package], color=COLORS[package], ms=4, lw=1.5, label=LABELS[package])
        ax.set_title(title, loc='left')
        ax.set_xlabel('k-points')
    axes[0].set_ylabel('time (s)')
    axes[0].legend(frameon=False)
    ax = axes[3]
    for package in ('tbkit', 'kwant'):
        rows = sorted((r for r in data['transport'] if r['package'] == package), key=lambda r: r['sites'])
        ax.loglog([r['sites'] for r in rows], [r['seconds'] for r in rows], STYLES[package],
                  marker=MARKERS[package], color=COLORS[package], ms=4, lw=1.5, label=LABELS[package])
    ax.set_title('S-matrix, $W\\times W$', loc='left')
    ax.set_xlabel('sites')
    ax.legend(frameon=False)
    for letter, ax in zip('abcd', axes):
        ax.grid(which='major')
        ax.set_title(f'({letter}) ' + ax.get_title(loc='left'), loc='left')
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_benchmarks.pdf')
    plt.close(fig)


def tex(s):
    """Escape text for LaTeX, leaving $...$ math untouched."""
    parts = s.split('$')
    for n in range(0, len(parts), 2):
        parts[n] = (parts[n].replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%')
                    .replace('&', r'\&').replace('^', r'\^{}'))
    return '$'.join(parts)


def tables():
    data = json.loads((RESULTS / 'validation.json').read_text())
    lines = []
    for ref, checks in data['cross_checks'].items():
        for name, dev in checks.items():
            lines.append(f'{LABELS[ref]} & {tex(name)} & \\num{{{dev:.1e}}} \\\\')
    (TABLES / 'crosschecks.tex').write_text('\n'.join(lines) + '\n')
    lines = [f"{tex(e['result'])} & {tex(e['reference'])} & \\url{{{e['example']}}} & "
             f"{tex(e['tolerance'])} \\\\" for e in data['analytic']]
    (TABLES / 'analytic.tex').write_text('\n'.join(lines) + '\n')
    if (RESULTS / 'benchmarks.json').exists():
        bench = json.loads((RESULTS / 'benchmarks.json').read_text())
        lines = []
        for width in sorted({r['width'] for r in bench['transport']}):
            a, b = (next(r for r in bench['transport'] if r['width'] == width and r['package'] == p)
                    for p in ('tbkit', 'kwant'))
            lines.append(f"{width**2:,} & {a['seconds']:.3g} & {a['peak_mb']:.0f} & "
                         f"{b['seconds']:.3g} & {b['peak_mb']:.0f} \\\\".replace(',', '{,}'))
        (TABLES / 'bench_transport.tex').write_text('\n'.join(lines) + '\n')
        norb, nk = max(r['norb'] for r in bench['kspace']), max(r['nk'] for r in bench['kspace'])
        names = {'bands': 'band energies', 'chern': 'Chern number', 'wilson': 'Wilson-loop flow'}
        lines = []
        for task, name in names.items():
            a, b = (next(r for r in bench['kspace'] if r['task'] == task and r['norb'] == norb
                         and r['nk'] == nk and r['package'] == p) for p in ('tbkit', 'pythtb'))
            lines.append(f"{name} & {a['seconds']:.3g} & {a['peak_mb']:.0f} & "
                         f"{b['seconds']:.3g} & {b['peak_mb']:.0f} \\\\")
        (TABLES / 'bench_kspace.tex').write_text('\n'.join(lines) + '\n')
        (TABLES / 'bench_setup.tex').write_text(
            f"\\newcommand{{\\benchnorb}}{{{norb}}}\n\\newcommand{{\\benchnk}}{{{nk}}}\n"
            f"\\newcommand{{\\benchplatform}}{{{tex(bench['platform'])}}}\n")
    v = data['versions']
    (TABLES / 'versions.tex').write_text(
        f"\\newcommand{{\\tbkitversion}}{{{v['tbkit']}}}\n"
        f"\\newcommand{{\\kwantversion}}{{{v['kwant']}}}\n"
        f"\\newcommand{{\\pythtbversion}}{{{v['pythtb']}}}\n"
        f"\\newcommand{{\\numpyversion}}{{{v['numpy']}}}\n"
        f"\\newcommand{{\\scipyversion}}{{{v['scipy']}}}\n")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    for name in ('models', 'topology', 'response', 'driven'):
        panels(name)
        print(f'fig_{name}.pdf')
    validation()
    print('fig_validation.pdf')
    if (RESULTS / 'benchmarks.json').exists():
        benchmarks()
        print('fig_benchmarks.pdf')
    tables()
    print('tables/')


if __name__ == '__main__':
    main()
