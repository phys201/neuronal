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
        sample = psp_fit(data, 1000, initial_guess, plot=False)
        summary = pm.summary(sample)['mean']
        b = summary['b']
        a1 = summary['a1']
        sigma = summary['sigma']
        t1 = summary['t1']
        tau_d1 = summary['tau_d1']
        tau_r1 = summary['tau_r1']
        self.assertTrue((b < -30.38) and (b > -30.42))
        self.assertTrue((t1 < 451.44) and (t1 > 451.42))
        self.assertTrue((sigma < 0.024) and (sigma > 0.023))
