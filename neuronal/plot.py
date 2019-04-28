"""
Convenient functions for plotting data and best fit.

Authors: Amelia Paine, Han Sae Jung
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm


def plot_fit(data, sample):
    """
    Plots data and best fit from parameter estimation

    Parameters
    ----------
    data : NeuronalData
        Imported data
    sample : pymc3.backends.base.MultiTrace
        Result from pymc3 calculation
    """
    t = np.array(data.data['T'])
    v = np.array(data.data['V'])
    summary = pm.summary(sample)
    if data.num_psp == 1:
        [b_start, b0, b_end, a0, sigma, t_psp0, tau_d0, tau_r0] = summary_test['mean']
        t_psp = np.array([t_psp0])
        b = np.array([b0])
        a = np.array([a0])
        tau_d = np.array([tau_d0])
        tau_r = np.array([tau_r0])
        
    elif data.num_psp == 2:
        [b_start, b0, b1, b_end, a0, a1, sigma, t_psp0, t_psp1, tau_d0, tau_d1, tau_r0, tau_r1] = summary_test['mean']
        t_psp = np.array([t_psp0, t_psp1])
        b = np.array([b0, b1])
        a = np.array([a0, a1])
        tau_d = np.array([tau_d0, tau_d1])
        tau_r = np.array([tau_r0, tau_r1])

    elif data.num_psp == 3:
        [b_start, b0, b1, b2, b_end, a0, a1, a2, sigma, 
         t_psp0, t_psp1, t_psp2, tau_d0, tau_d1, tau_d2, tau_r0, tau_r1, tau_r2] = summary_test['mean']
        t_psp = np.array([t_psp0, t_psp1, t_psp2])
        b = np.array([b0, b1, b2])
        a = np.array([a0, a1, a2])
        tau_d = np.array([tau_d0, tau_d1, tau_d2])
        tau_r = np.array([tau_r0, tau_r1, tau_r2])
        
    model = (t <= t_psp[0]) * (b_start + (b[0] - b_start) / (t_psp[0] - t[0]) * (t - t[0])) +\
            np.sum([
                    (t >= t_psp[i]) * (a[i] * (np.exp(-(t-t_psp[i]) / tau_d[i]) - np.exp(-(t-t_psp[i]) / tau_r[i])) +\
                    (t <= t_psp[i+1]) * (b[i] + (b[i+1] - b[i]) / (t_psp[i+1] - t_psp[i]) * (t - t_psp[i])))
                    for i in range(num_psp - 1)], axis=0) +\
            (t >= t_psp[-1]) * (a[-1] * (np.exp(-(t-t_psp[-1]) / tau_d[-1]) - np.exp(-(t-t_psp[-1]) / tau_r[-1])) +\
            (b[-1] + (b_end - b[-1]) / (t[-1] - t_psp[-1]) * (t - t_psp[-1])))
    plt.plot(t, v)
    plt.plot(t, model, c='r')
    plt.title('Estimated Fit from the Model')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.show()
