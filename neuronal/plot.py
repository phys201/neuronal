"""
Convenient functions for plotting data and best fit.

Authors: Amelia Paine, Han Sae Jung
"""

import matplotlib.pyplot as plt
from .model import psp_model
from .analysis import *


def get_params_from_summary(data, summary):
    """
        Returns mean model parameters, grouped into numpy arrays when relevant

        Parameters
        ----------
        data : NeuronalData
            Imported data
        summary : Pandas DataFrame
            Summary of the result from pymc3 calculation

        Returns
        -------
        sigma : float
        b_start : float
        b_end : float
        b : list of float
        a : list of float
        t_psp : list of float
        tau_d : list of float
        tau_r : list of float
    """
    means = summary['mean']
    num_psp = data.num_psp
    b_start = means['b_start']
    b_end = means['b_end']
    sigma = means['sigma']
    b = [means['b__' + str(i)] for i in range(num_psp)]
    a = [means['a__' + str(i)] for i in range(num_psp)]
    t_psp = [means['t_psp__' + str(i)] for i in range(num_psp)]
    tau_d = [means['tau_d__' + str(i)] for i in range(num_psp)]
    tau_r = [means['tau_r__' + str(i)] for i in range(num_psp)]
    return sigma, b_start, b_end, b, a, t_psp, tau_d, tau_r



