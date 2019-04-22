from unittest import TestCase
import os
import numpy as np
import pandas as pd
from neuronal.io import get_example_data_file_path, NeuronalData
from neuronal.model import *

class TestModel(TestCase):
    def test_single_psp(self):
        example_file_path = get_example_data_file_path('single_PSP_data.txt')
        data = NeuronalData(example_file_path)
        initial_guess = {
                         'b': -30.39, 
                         'a1': 0.3, 
                         't1': 451.43, 
                         'tau_d1': 0.01, 
                         'tau_r1': 0.001
                        }
        sample = psp_fit(data, 5, initial_guess, plot=False, seed=42, tune=10)
        summary = pm.summary(sample)['mean']
        b = summary['b']
        a1 = summary['a1']
        sigma = summary['sigma']
        t1 = summary['t1']
        tau_d1 = summary['tau_d1']
        tau_r1 = summary['tau_r1']
        print(b, a1, sigma, t1)
        self.assertAlmostEqual(b, -30.387490320942778)
        self.assertAlmostEqual(a1, 0.30088890597428075)
        self.assertAlmostEqual(sigma, 0.80651278277974)
        self.assertAlmostEqual(t1, 451.43077112986083)
