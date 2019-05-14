from unittest import TestCase
import os
import pandas as pd
import numpy as np
from neuronal.io import get_example_data_file_path, NeuronalData


class TestIo(TestCase):

    def test_data_io(self):
        example_file_path = get_example_data_file_path('single_PSP_data.txt')
        self.assertTrue(os.path.exists(example_file_path))
        data = NeuronalData(example_file_path)
        self.assertTrue(isinstance(data.data, pd.DataFrame))
        self.assertTrue(isinstance(data.data['T'][0], float))
        self.assertAlmostEqual(data.data['T'][0], 451.4001062134890390)
        self.assertTrue(isinstance(data.data['V'][0], float))
        self.assertAlmostEqual(data.data['V'][0], -30.38982152938842773)
        self.assertTrue(len(data.data['T']) == 942)
        self.assertTrue(data.num_psp == 1)

    def test_decimate(self):
        example_file_path = get_example_data_file_path('single_PSP_data.txt')
        data = NeuronalData(example_file_path)
        decimated = data.decimate(10)
        self.assertTrue(isinstance(decimated, NeuronalData))
        self.assertTrue(len(decimated.data['T']) == 95)
        self.assertAlmostEqual(decimated.data['T'][0], 451.4001062134890390)

    def test_randomly_reduce(self):
        example_file_path = get_example_data_file_path('single_PSP_data.txt')
        data = NeuronalData(example_file_path)
        np.random.seed(42)
        rand_reduced = data.randomly_reduce(10).data
        self.assertTrue(len(rand_reduced) == 94)
        self.assertTrue(list(rand_reduced.index.values)[0] == 23)
        self.assertAlmostEqual(rand_reduced['T'].values[0], 451.4025491237387)
        self.assertAlmostEqual(rand_reduced['V'].values[0], -30.4366668065389)

    def test_num_psp(self):
        example_file_path = get_example_data_file_path('two_PSP_data.txt')
        self.assertTrue(os.path.exists(example_file_path))
        data = NeuronalData(example_file_path, num_psp=2)
        self.assertTrue(isinstance(data.data, pd.DataFrame))
        self.assertTrue(isinstance(data.data['T'][0], float))
        self.assertAlmostEqual(data.data['T'][0], 422.5000531067445309)
        self.assertTrue(isinstance(data.data['V'][0], float))
        self.assertAlmostEqual(data.data['V'][0], -32.16990232467651367)
        self.assertTrue(len(data.data['T']) == 4709)
        self.assertTrue(data.num_psp == 2)

    def test_constructor_exceptions(self):
        self.assertRaises(Exception, NeuronalData, 0)
        self.assertRaises(Exception, NeuronalData, pd.DataFrame())