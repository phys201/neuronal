"""
Convenient functions for plotting data and best fit.

Authors: Amelia Paine, Han Sae Jung
"""

import numpy as np
import matplotlib.pyplot as plt


def get_params_from_summary(data, summary):
    """
        Returns mean model parameters, grouped into numpy arrays when relevant

        Parameters
        ----------
        data : NeuronalData
            Imported data
        summary : Pandas DataFrame
            Summary of the result from pymc3 calculation
    """
    means = summary['mean']
    num_psp = data.num_psp
    b_start = means['b_start']
    b_end = means['b_end']
    sigma = means['sigma']
    b = np.array([means['b__' + str(i)] for i in range(num_psp)])
    a = np.array([means['a__' + str(i)] for i in range(num_psp)])
    t_psp = np.array([means['t_psp__' + str(i)] for i in range(num_psp)])
    tau_d = np.array([means['tau_d__' + str(i)] for i in range(num_psp)])
    tau_r = np.array([means['tau_r__' + str(i)] for i in range(num_psp)])
    return sigma, b_start, b_end, b, a, t_psp, tau_d, tau_r


def psp_model_for_plot(data, summary):
    """
    Returns a psp_model for a given summary calculated from a pymc3 calculation 

    Parameters
    ----------
    data : NeuronalData
        Imported data
    summary : Pandas DataFrame
        Summary of the result from pymc3 calculation
    """
    t = np.array(data.data['T'])
    v = np.array(data.data['V'])
    num_psp = data.num_psp
    sigma, b_start, b_end, b, a, t_psp, tau_d, tau_r = get_params_from_summary(data, summary)
        
    model = (t <= t_psp[0]) * (b_start + (b[0] - b_start) / (t_psp[0] - t[0]) * (t - t[0])) +\
            np.sum([
                    (t >= t_psp[i]) * (a[i] * (np.exp(-(t-t_psp[i]) / tau_d[i]) - np.exp(-(t-t_psp[i]) / tau_r[i])) +\
                    (t <= t_psp[i+1]) * (b[i] + (b[i+1] - b[i]) / (t_psp[i+1] - t_psp[i]) * (t - t_psp[i])))
                    for i in range(num_psp - 1)], axis=0) +\
            (t >= t_psp[-1]) * (a[-1] * (np.exp(-(t-t_psp[-1]) / tau_d[-1]) - np.exp(-(t-t_psp[-1]) / tau_r[-1])) +\
            (b[-1] + (b_end - b[-1]) / (t[-1] - t_psp[-1]) * (t - t_psp[-1])))
    return model


def plot_fit(data, summary, show_plot=True):
    """
    Plots the data and the best fit from psp_model using a summary calculated from pymc3 and
    returns 'matplotlib.axes._subplots.AxesSubplot' object

    Parameters
    ----------
    data : NeuronalData
        Imported data
    summary : Pandas DataFrame
        Summary of the result from pymc3 calculation
    show_plot : bool
        If True, function will call plt.show() to display plot
    """
    t = np.array(data.data['T'])
    v = np.array(data.data['V'])
    model = psp_model_for_plot(data, summary)
    
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
