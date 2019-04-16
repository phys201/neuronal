import os
import numpy as np
import pandas as pd

class NeuronalData:
    def __init__(self, data_file, num_psp=1):
         self.data = load_data(data_file)
         self.num_psp = num_psp

def get_example_data_file_path(filename, data_dir='data'):
    '''
    Gets file path of example data file (in neuronal/data/ or other
    specified directory), independent of operating system.

    -- Parameters --
    filename: (string) name of example data file
    data_dir: (string, opt) directory containing example data

    -- Returns --
    (string) absolute path of example data file
    '''
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    data_dir = os.path.join(start_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)

def load_data(data_file):
    """
    Load data into a DataFrame
    
    Parameters
    ----------
    filename: (string) file containing space separated data
    """
    data_array = np.loadtxt(data_file)
    return pd.DataFrame(data=data_array.T, columns=['T', 'V'])
