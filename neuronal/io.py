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
    def __init__(self, data_source, num_psp=1):
        """
        Initialize NeuronalData object

        Data file should be space-separated values. First row is time (in seconds) and second row is signal
        (in millivolts).

        If data is provided as a DataFrame, it should have columns 'T' and 'V'.

        Parameters
        ----------
        data_source : str or pandas.DataFrame
            path to correctly formatted data (if string)
            DataFrame containing voltage vs. time data
        num_psp : int, optional
            number of PSPs in data
        """
        if isinstance(data_source, str):
            self.data = load_data(data_source)
        elif isinstance(data_source, pd.DataFrame):
            if (len(data_source.columns.values) == 2
                    and 'T' in data_source.columns.values and 'V' in data_source.columns.values):
                self.data = data_source
            else:
                raise Exception("Source DataFrame columns are incorrect (should be 'T' and 'V')")
        else:
            raise Exception('Initialization error. data_source is the wrong type (should be str or pandas DataFrame) : '
                            + str(type(data_source)))
        self.num_psp = num_psp

    def decimate(self, n):
        """
        Returns a NeuronalData object with every nth data point

        Parameters
        ----------
        n : int
            factor by which to reduce number of data points

        Returns
        -------
        NeuronalData
            New NeuronalData object containing decimated data
        """
        return NeuronalData(self.data.iloc[::n, :], num_psp=self.num_psp)

    def randomly_reduce(self, n, replace=False, seed=None):
        """
        Returns a NeuronalData object with number of data points randomly reduced by a factor of n.
        Behavior is similar to decimate except the points are random.

        Parameters
        ----------
        n : float
            Factor by which to reduce number of data points
        replace : bool, optional
            Sample with replacement
        seed : int, optional
            Seed for random number generator

        Returns
        -------
        NeuronalData
            New NeuronalData object containing randomly sampled data points
        """
        df = self.data.sample(frac=1.0/n, replace=replace, random_state=seed)
        return NeuronalData(df.sort_values(by=['T']), num_psp=self.num_psp)


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
