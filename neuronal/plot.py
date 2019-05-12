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


def plot_fit(data, df, show_plot=True):
    """
    Plots the data and the best fit from psp_model using the pymc3 trace and
    returns 'matplotlib.axes._subplots.AxesSubplot' object

    Parameters
    ----------
    data : NeuronalData
        Imported data
    df : pandas.DataFrame
        Dataframe containing the results from pymc3 calculation
    show_plot : bool, optional
        If True, function will call plt.show() to display plot

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes of plot
    """
    quantile_df = get_quantiles(df)
    t = np.array(data.data['T'])
    v = np.array(data.data['V'])
    sigma, b_start, b_end, b, a, t_psp, tau_d, tau_r = get_params_from_df(data, quantile_df)
    model = psp_model(data, b_start, b, b_end, a, t_psp, tau_d, tau_r)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(t, v)
    ax.plot(t, model, c='r')
    ax.set_title('Estimated Fit from the Model')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (mV)')
    if show_plot:
        plt.show()
    return ax
