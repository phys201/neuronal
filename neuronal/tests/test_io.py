from unittest import TestCase
import os
import numpy as np
import pandas as pd
from neuronal.io import get_example_data_file_path, load_data, NeuronalData

class TestIo(TestCase):
    def test_data_io(self):
        example_file_path = get_example_data_file_path('single_PSP_data.txt')
        self.assertTrue(os.path.exists(example_file_path))
        data = NeuronalData(example_file_path)
        self.assertTrue(isinstance(data.data, pd.DataFrame))
        self.assertTrue(isinstance(data.data['T'][0], float))
        self.assertTrue(isinstance(data.data['V'][0], float))
        self.assertTrue(data.num_psp == 1)
