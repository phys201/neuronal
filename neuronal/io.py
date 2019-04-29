"""
This module governs the import of experimental data.

Authors: Amelia Paine, Han Sae Jung
"""

import os
import numpy as np
import pandas as pd


class NeuronalData:
    """
    An object to hold imported neuronal data and metadata
    """
    def __init__(self, data_file, num_psp=1):
        """
        Initialize NeuronalData object

        Data should be space-separated values. First row is time (in seconds) and second row is signal (in millivolts).

        Parameters
        ----------
        data_file : str
            path to correctly formatted data
        num_psp : int, optional
            number of PSPs in data
        """
        self.data = load_data(data_file)
        self.num_psp = num_psp


def get_example_data_file_path(filename, data_dir='data'):
    """
    Gets file path of example data file (in neuronal/data/ or other
    specified directory), independent of operating system.

    Parameters
    ----------
    filename : str
        name of example data file
    data_dir : str, optional
        directory containing example data

    Returns
    -------
    str
        absolute path of example data file
    """
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    data_dir = os.path.join(start_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)


def load_data(data_file):
    """
    Load data into a DataFrame
    
    Parameters
    ----------
    data_file : str
        file containing space separated data

    Returns
    -------
    DataFrame
        loaded data organized into time (T) and voltage (V)
    """
    data_array = np.loadtxt(data_file)
    return pd.DataFrame(data=data_array.T, columns=['T', 'V'])
