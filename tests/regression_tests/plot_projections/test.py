from tests.regression_tests import config
from tests.testing_harness import PlotTestHarness


def test_plot():
    harness = PlotTestHarness(('plot_1.png', 'example1.png', 'example2.png', 'example3.png', 'orthographic_example1.png'))
    harness.main()
