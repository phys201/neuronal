from unittest import TestCase
from neuronal.io import get_example_data_file_path
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
        self.assertAlmostEqual(sigma, 0.5251100153998798)
        self.assertAlmostEqual(b_start, -30.372609326154524)
        self.assertAlmostEqual(b_end, -30.36056833174976)
        self.assertAlmostEqual(b[0], -30.398952468916747)
        self.assertAlmostEqual(a[0], 0.2915479489155847)
        self.assertAlmostEqual(t_psp[0], 451.4294192155971)
        self.assertAlmostEqual(tau_d[0], 0.009691246075731536)
        self.assertAlmostEqual(tau_r[0], 0.0009123478585051712)

    def test_peak_amplitudes(self):
        df = calculate_peak_amplitudes(0, sample, np.max(data.data['T']), data.num_psp)
        self.assertTrue('peak_amp__0' in df.columns.values)
        self.assertAlmostEqual(df['peak_amp__0'][0], 0.17820178467355885)
        self.assertRaises(KeyError, calculate_peak_amplitudes, 1, sample, np.max(data.data['T']), data.num_psp)

    def test_report(self):
        value_minus, value_mid, value_plus = report_best_fit(get_quantiles(sample), 'b__0', print_fit=True)
        self.assertAlmostEqual(value_minus, -30.46566738900276)
        self.assertAlmostEqual(value_mid, -30.398952468916747)
