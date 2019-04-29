from unittest import TestCase
import os
import pandas as pd
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

    def test_data_io_2(self):
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

    def test_data_io_3(self):
        example_file_path = get_example_data_file_path('three_PSP_data.txt')
        self.assertTrue(os.path.exists(example_file_path))
        data = NeuronalData(example_file_path, num_psp=3)
        self.assertTrue(isinstance(data.data, pd.DataFrame))
        self.assertTrue(isinstance(data.data['T'][0], float))
        self.assertAlmostEqual(data.data['T'][0], 450.7500796601167963)
        self.assertTrue(isinstance(data.data['V'][0], float))
        self.assertAlmostEqual(data.data['V'][0], -30.09314139684041578)
        self.assertTrue(len(data.data['T']) == 3061)
        self.assertTrue(data.num_psp == 3)