from unittest import TestCase
from neuronal.io import get_example_data_file_path, NeuronalData
from neuronal.model import *
import warnings
import pymc3 as pm

example_file_path = get_example_data_file_path('single_PSP_data.txt')
data = NeuronalData(example_file_path)


class TestModel(TestCase):
    def test_model(self):
        # TODO: test model with explicitly set parameters
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
        summary = pm.summary(sample)['mean']
        b_start = summary['b_start']
        a = summary['a__0']
        sigma = summary['sigma']
        t_psp = summary['t_psp__0']
        self.assertAlmostEqual(b_start, -30.39624270830434)
        self.assertAlmostEqual(a, 0.30565798385998183)
        self.assertAlmostEqual(sigma, 0.4352468747178174)
        self.assertAlmostEqual(t_psp, 451.4298875114434)

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
        summary = pm.summary(sample)['mean']
        b_start = summary['b_start']
        b = summary['b__0']
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