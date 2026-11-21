"""
Opt-in pytest plugin that flags assertions likely to break on another
platform or BLAS build:

* ``np.argmax``, ``np.argmin`` and ``np.argsort`` called directly from test
  code on floating-point data whose deciding values tie to within ``REL``
  (relative), so the winning index is decided by rounding;
* ``assertEqual`` comparing non-integer floats or complex numbers exactly.

Run it with::

    pytest tests/ --fragile

A hit is a candidate, not a bug: a tie is harmless when the test then
compares only values (e.g. sorted degenerate eigenvalues), and an exact
float comparison is harmless for parsed constants or integer sums. It sees
only ties present on the machine it runs on, and not array-method calls
such as ``x.argmin()``, which cannot be patched.
"""
import collections
import json
import os
import sys
import unittest

import numpy as np

REL = 1e-9
_NAMES = ('argmax', 'argmin', 'argsort')
_orig = {name: getattr(np, name) for name in _NAMES}
_orig_assert_equal = unittest.TestCase.assertEqual
_out = None


def _site():
    # the direct caller of the patched function, if it is test code
    code = sys._getframe(2).f_code
    if os.sep + 'tests' + os.sep not in code.co_filename:
        return None
    return f'{os.path.basename(code.co_filename)}:{sys._getframe(2).f_lineno}'


def _log(kind, site, detail):
    with open(_out, 'a') as fh:
        fh.write(json.dumps({'kind': kind, 'site': site, 'detail': detail}) + '\n')


def _tie(a, axis, name):
    '''Smallest relative gap between the deciding values, if below REL.'''
    a = np.asarray(a)
    if a.size < 2 or not np.issubdtype(a.dtype, np.floating):
        return None
    a = a.reshape(1, -1) if axis is None else np.moveaxis(a, axis, -1).reshape(-1, a.shape[axis])
    s = np.sort(a, axis=-1)
    if name == 'argmax':
        gap = s[:, -1] - s[:, -2]
    elif name == 'argmin':
        gap = s[:, 1] - s[:, 0]
    else:  # argsort: the closest distinct neighbours
        d = np.diff(s, axis=-1)
        gap = np.where(d > 0, d, np.inf).min(axis=-1)
    rel = (gap / np.maximum(np.abs(s).max(axis=-1), 1e-300)).min()
    return float(rel) if rel < REL else None


def _wrap(name):
    orig = _orig[name]

    def probe(a, axis=-1 if name == 'argsort' else None, *args, **kwargs):
        site = _site()
        if site:
            try:
                gap = _tie(a, axis, name)
            except Exception:
                gap = None
            if gap is not None:
                _log(name, site, gap)
        return orig(a, axis, *args, **kwargs)
    return probe


def _assert_equal(self, first, second, msg=None):
    inexact = (float, complex, np.inexact)
    if isinstance(first, inexact) or isinstance(second, inexact):
        try:  # integer values (0., 2., 1j) are exact on every platform
            z = complex(first)
            integral = z.real == round(z.real) and z.imag == round(z.imag)
        except (TypeError, ValueError, OverflowError):
            integral = False
        site = None if integral else _site()
        if site:
            _log('assertEqual', site, f'{first!r} vs {second!r}')
    return _orig_assert_equal(self, first, second, msg)


def pytest_configure(config):
    global _out
    _out = str(config.rootpath / '.pytest_cache' / 'fragile_probe.jsonl')
    if not hasattr(config, 'workerinput'):  # the controller (or a run without xdist)
        os.makedirs(os.path.dirname(_out), exist_ok=True)
        open(_out, 'w').close()
    for name in _NAMES:
        setattr(np, name, _wrap(name))
    unittest.TestCase.assertEqual = _assert_equal


def pytest_unconfigure(config):
    for name in _NAMES:
        setattr(np, name, _orig[name])
    unittest.TestCase.assertEqual = _orig_assert_equal


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if hasattr(config, 'workerinput') or not os.path.exists(_out):
        return
    hits = collections.defaultdict(list)
    with open(_out) as fh:
        for line in fh:
            row = json.loads(line)
            hits[(row['kind'], row['site'])].append(row['detail'])
    tr = terminalreporter
    tr.section('fragile probe')
    if not hits:
        tr.write_line('no near-ties or exact float comparisons in test code')
        return
    for (kind, site), details in sorted(hits.items(), key=lambda h: h[0][1]):
        if kind == 'assertEqual':
            tr.write_line(f'{site}: exact float assertEqual, e.g. {details[0]}')
        else:
            tr.write_line(f'{site}: {kind} near-tie, relative gap {min(details):.1e}')
    tr.write_line(f'{len(hits)} candidate(s); details in {_out}')
