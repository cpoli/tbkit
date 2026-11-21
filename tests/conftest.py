import matplotlib
matplotlib.use('Agg')


def pytest_addoption(parser):
    parser.addoption('--fragile', action='store_true',
                     help='flag test-side near-ties and exact float comparisons (tests/fragile_probe.py)')


def pytest_configure(config):
    if config.getoption('--fragile'):
        from tests import fragile_probe
        config.pluginmanager.register(fragile_probe, 'fragile_probe')
