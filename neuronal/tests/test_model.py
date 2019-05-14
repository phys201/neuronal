from unittest import TestCase
from neuronal.io import get_example_data_file_path, NeuronalData
from neuronal.model import *
from neuronal.analysis import *
import warnings

example_file_path = get_example_data_file_path('single_PSP_data.txt')
data = NeuronalData(example_file_path)


class TestModel(TestCase):
    def test_model(self):
        model = psp_model(data, -30.397341, [-30.397341], -30.397341, [0.290294],
                          [451.427934], [0.015988], [0.002611])
        self.assertTrue(len(model) == 942)
        self.assertAlmostEqual(model[0], -30.397341)

    def test_likelihood(self):
        log_likelihood = psp_log_likelihood(data, -30.397341, [-30.397341], -30.397341, 0.023289, [0.290294],
                                            [451.427934], [0.015988], [0.002611])
        self.assertAlmostEqual(log_likelihood, 2208.258912268257)

    def test_psp_fit(self):
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
        self.assertTrue(len(sample['b_start']) == 10)
        b_start = sample['b_start'][0]
        a = sample['a__0'][0]
        sigma = sample['sigma'][0]
        t_psp = sample['t_psp__0'][0]
        self.assertAlmostEqual(b_start, -30.351857720030377)
        self.assertAlmostEqual(a, 0.2555859708643207)
        self.assertAlmostEqual(sigma, 0.4811457182253871)
        self.assertAlmostEqual(t_psp, 451.42915617432124)

    def test_user_input_priors(self):
        initial_guess = {
            'b_start': -29,
            'b_end': -29,
            'b': [-29],
            'sigma': 0.3,
            'a': [0.3],
            't_psp': [451.43],
            'tau_d': [0.01],
            'tau_r': [0.001]
        }
        sample = psp_fit(data, 5, initial_guess, seed=42, tune=10, prior_ranges={'foo': 0,
                                                                                 'b_start': (-30, -20),
                                                                                 'b': (-30, -20)})
        b_start = np.mean(sample['b_start'])
        b = np.mean(sample['b__0'])
        self.assertTrue(-20 > b_start > -30)
        self.assertTrue(-20 > b > -30)

    def test_validate_params(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_params(data, {'sigma': 0.5, 'b_start': -30.39, 'b_end': -30.39, 'b': [-30.39], 'a': [0.3],
                                   't_psp': [451.43], 'tau_d': [0.01], 'tau_r': [0.001]})
            self.assertTrue(len(w) == 0)
            validate_params(data, {'b_start': -30.39, 'b_end': -30.39, 'b': [-30.39], 'a': [0.3],
                                   't_psp': [451.43], 'tau_d': [0.01], 'tau_r': [0.001]})
            validate_params(data, {'sigma': 0.5, 'b_start': -30.39, 'b_end': -30.39, 'b': [-30.39], 'a': [0.3],
                                   't_psp': [451.43], 'tau_d': [0.01], 'tau_r': [0.001], 'foo': 0})
            validate_params(data, {'sigma': 0.5, 'b_start': -30.39, 'b_end': -30.39, 'b': [-30.39, 0], 'a': [0.3],
                                   't_psp': [451.43], 'tau_d': [0.01], 'tau_r': [0.001]})
            validate_params(data, {'sigma': 0.5, 'b_start': -30.39, 'b_end': -30.39, 'b': -30.39, 'a': [0.3],
                                   't_psp': [451.43], 'tau_d': [0.01], 'tau_r': [0.001]})
            self.assertTrue(len(w) == 4)

    def test_simulated_data(self):
        np.random.seed(42)
        sim = simulate_psp_data(1., 0., [0.], 0., [0.5], [1.], [0.01], [0.001], (0., 5), 0.1)
        self.assertTrue(len(sim) == 50)
        self.assertAlmostEqual(sim['T'][0], 0.)
        self.assertAlmostEqual(sim['T'][49], 4.9)
        self.assertAlmostEqual(sim['V'][0], 0.4967141530112327)

