from unittest import TestCase
from neuronal.io import get_example_data_file_path, NeuronalData
from neuronal.model import *
from neuronal.analysis import *
import matplotlib

example_file_path = get_example_data_file_path('single_PSP_data.txt')
data = NeuronalData(example_file_path)
initial_guess = {
    'b_start': -30.39,
    'b_end': -30.39,
    'b': [-30.39],
    'a': [0.3],
    't_psp': [451.43],
    'tau_d': [0.01],
    'tau_r': [0.001]
}
sample = psp_fit(data, 5, initial_guess, seed=42, tune=10, suppress_warnings=True)


class TestAnalysis(TestCase):
    def test_plot(self):
        ax = plot_fit(data, sample, show_plot=False)
        self.assertTrue(isinstance(ax, matplotlib.axes._base._AxesBase))

    def test_get_params(self):
        sigma, b_start, b_end, b, a, t_psp, tau_d, tau_r = get_params_from_df(data, sample)

    def test_peak_amplitudes(self):
        df = calculate_peak_amplitudes(0, sample, np.max(data.data['T']), data.num_psp)

    def test_report(self):
        value_minus, value_mid, value_plus = report_best_fit(get_quantiles(sample), 'b', print_fit=True)
