    
from unittest import TestCase
from neuronal import get_data_file_path, loadtxt
import numpy as np

class TestIo(TestCase):
    def test_io(self):
        data = loadtxt(get_data_file_path('single_PSP_data.txt'))
        assert data.shape = (2, 942)
