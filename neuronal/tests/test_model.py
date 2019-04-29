from unittest import TestCase
from neuronal.io import get_example_data_file_path, NeuronalData
from neuronal.model import *
import warnings


class TestModel(TestCase):
    def test_model(self):
        example_file_path = get_example_data_file_path('single_PSP_data.txt')
        data = NeuronalData(example_file_path)
        model = psp_model(data, -30.397341, [-30.397341], -30.397341, [0.290294],
                          [451.427934], [0.015988], [0.002611])
        self.assertTrue(len(model) == 942)
        self.assertAlmostEqual(model[0], -30.397341)

    def test_likelihood(self):
        example_file_path = get_example_data_file_path('single_PSP_data.txt')
        data = NeuronalData(example_file_path)
        log_likelihood = psp_log_likelihood(data, -30.397341, [-30.397341], -30.397341, 0.023289, [0.290294],
                                            [451.427934], [0.015988], [0.002611])
        self.assertAlmostEqual(log_likelihood, 2208.258912268257)

    def test_psp_fit(self):
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
        summary = pm.summary(sample)['mean']
        b_start = summary['b_start']
        a = summary['a__0']
        sigma = summary['sigma']
        t_psp = summary['t_psp__0']
        self.assertAlmostEqual(b_start, -30.41177768455343)
        self.assertAlmostEqual(a, 0.30139712310721395)
        self.assertAlmostEqual(sigma, 0.484764558519905)
        self.assertAlmostEqual(t_psp, 451.42953770165485)

    def test_validate_params(self):
        warnings.filterwarnings('error')
        example_file_path = get_example_data_file_path('single_PSP_data.txt')
        data = NeuronalData(example_file_path)
        try:  # Should have no warnings
            validate_params(data, {'sigma': 0.5, 'b_start': -30.39, 'b_end': -30.39, 'b': [-30.39], 'a': [0.3],
                                   't_psp': [451.43], 'tau_d': [0.01], 'tau_r': [0.001]})
        except Warning:
            self.assertTrue(False)
        try:  # Missing param
            warn = False
            validate_params(data, {'b_start': -30.39, 'b_end': -30.39, 'b': [-30.39], 'a': [0.3],
                                   't_psp': [451.43], 'tau_d': [0.01], 'tau_r': [0.001]})
        except Warning:
            warn = True
        finally:
            self.assertTrue(warn)
        try:  # Unexpected param
            warn = False
            validate_params(data, {'sigma': 0.5, 'b_start': -30.39, 'b_end': -30.39, 'b': [-30.39], 'a': [0.3],
                                   't_psp': [451.43], 'tau_d': [0.01], 'tau_r': [0.001], 'foo': 0})
        except Warning:
            warn = True
        finally:
            self.assertTrue(warn)
        try:  # Incorrect dimension
            warn = False
            validate_params(data, {'sigma': 0.5, 'b_start': -30.39, 'b_end': -30.39, 'b': [-30.39, 0], 'a': [0.3],
                                   't_psp': [451.43], 'tau_d': [0.01], 'tau_r': [0.001]})
        except Warning:
            warn = True
        finally:
            self.assertTrue(warn)
        try:  # Incorrect parameter type
            warn = False
            validate_params(data, {'sigma': 0.5, 'b_start': -30.39, 'b_end': -30.39, 'b': -30.39, 'a': [0.3],
                                   't_psp': [451.43], 'tau_d': [0.01], 'tau_r': [0.001]})
        except Warning:
            warn = True
        finally:
            self.assertTrue(warn)