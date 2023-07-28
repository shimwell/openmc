from tests.regression_tests import config
from tests.testing_harness import PlotTestHarness


def test_plot_overlap():
    harness = PlotTestHarness(('plot_1.png', 'plot_2.png', 'plot_3.png',
                               'plot_4.h5'))
    harness.main()
