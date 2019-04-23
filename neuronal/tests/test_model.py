from unittest import TestCase
from neuronal.io import get_example_data_file_path, NeuronalData
from neuronal.model import *


class TestModel(TestCase):
    def test_likelihood(self):
        example_file_path = get_example_data_file_path('single_PSP_data.txt')
        data = NeuronalData(example_file_path)
        log_likelihood = psp_log_likelihood(data, -30.397341, 0.023289, [0.290294], [451.427934], [0.015988], [0.002611])
        self.assertAlmostEqual(log_likelihood.eval(), 2208.258912268257)

    def test_psp_fit(self):
        example_file_path = get_example_data_file_path('single_PSP_data.txt')
        data = NeuronalData(example_file_path)
        initial_guess = {
                         'b': -30.39, 
                         'a0': 0.3,
                         't_psp0': 451.43,
                         'tau_d0': 0.01,
                         'tau_r0': 0.001
                        }
        sample = psp_fit(data, 5, initial_guess, plot=False, seed=42, tune=10)
        summary = pm.summary(sample)['mean']
        b = summary['b']
        a1 = summary['a0']
        sigma = summary['sigma']
        t1 = summary['t_psp0']
        self.assertAlmostEqual(b, -30.387490320942778)
        self.assertAlmostEqual(a1, 0.30088890597428075)
        self.assertAlmostEqual(sigma, 0.80651278277974)
        self.assertAlmostEqual(t1, 451.43077112986083)

