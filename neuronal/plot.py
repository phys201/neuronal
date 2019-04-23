import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_fit(data, sample, psp1=False, psp3=False):
    """
    Plots a selected model
    Parameters
    ----------
    data : NeuronalData
        Imported data
    sample : pymc3.backends.base.MultiTrace
        Result from pymc3 calculation
    psp1 : boolean
        Model psp_fit (single psp signal)
    psp3 : boolean
        Model psp3_fit (three psp signals)
    """
    t = np.array(data.data['T'])
    v = np.array(data.data['V'])
    summary = pm.summary(sample)
    if psp1:
        b, a1, sigma, t1, tau_d1, tau_r1 = summary['mean']
        PSP_model = (t >= t1) * a1 * (np.exp(-(t-t1) / tau_d1) - np.exp(-(t-t1) / tau_r1)) + b
    elif psp3:
        b, a1, a2, a3, sigma, t1, tau_d1, tau_r1, t2, tau_d2, tau_r2, t3, tau_d3, tau_r3 = summary['mean']
        PSP_model = (t >= t1) * a1 * (tt.exp(-(t-t1) / tau_d1) - tt.exp(-(t-t1) / tau_r1)) + b +\
                    (t >= t2) * a2 * (tt.exp(-(t-t2) / tau_d2) - tt.exp(-(t-t2) / tau_r2)) +\
                    (t >= t3) * a3 * (tt.exp(-(t-t3) / tau_d3) - tt.exp(-(t-t3) / tau_r3))
    plt.plot(t, v)
    plt.plot(t, PSP_model, c='r')
    plt.title('Estimated Fit from the PSP Model')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.show()
