import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm

def plot_fit(data, sample):
    """
    Plots a selected model
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
        b, a0, sigma, t_psp0, tau_d0, tau_r0 = summary['mean']
        model = (t >= t_psp0) * a0 * (np.exp(-(t-t_psp0) / tau_d0) - np.exp(-(t-t_psp0) / tau_r0)) + b
    elif data.num_psp == 3:
        b, a0, a1, a2, sigma, t_psp0, tau_d0, tau_r0, t_psp1, tau_d1, tau_r1, t_psp2, tau_d2, tau_r2 = summary['mean']
        model = (t >= t_psp0) * a0 * (np.exp(-(t-t_psp0) / tau_d0) - np.exp(-(t-t_psp0) / tau_r0)) + b +\
                (t >= t_psp1) * a1 * (np.exp(-(t-t_psp1) / tau_d1) - np.exp(-(t-t_psp1) / tau_r1)) +\
                (t >= t_psp2) * a2 * (np.exp(-(t-t_psp2) / tau_d2) - np.exp(-(t-t_psp2) / tau_r2))
    plt.plot(t, v)
    plt.plot(t, model, c='r')
    plt.title('Estimated Fit from the Model')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.show()
