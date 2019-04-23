import theano.tensor as tt
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def psp_log_likelihood(data, b, sigma, a, t_psp, tau_d, tau_r):
    """
    Calculates log likelihood for a single psp assuming constant baseline
    
    Parameters
    ----------
    data : NeuronalData
        Experimental data
    b : float
        Constant baseline value
    sigma : float
        Width of Gaussian noise
    a : list of float
        amplitudes of PSPs
    t_psp : list of float
        PSP start times
    tau_d : list of float
        Decay constants (PSP long-range behavior)
    tau_r : lost of float
        Rise constants (PSP short-range behavior)

    Returns
    -------
    log_likelihood : float
        Log-likelihood of parameters
    """
    num_psp = data.num_psp
    t = np.array(data.data['T'])
    v = np.array(data.data['V'])

    # TODO: numpy broadcasting?
    model = b + np.sum(
        [(t >= t_psp[i]) * a[i] * (tt.exp(-(t - t_psp[i]) / tau_d[i]) - tt.exp(-(t - t_psp[i]) / tau_r[i]))
         for i in range(num_psp)])
    residual = ((v - model) / sigma) ** 2
    constant = 1 / np.sqrt(2 * np.pi * sigma ** 2)
    log_likelihood = (np.log(constant) - 0.5 * residual).sum()
    return log_likelihood

def psp_fit(data, nsamples, initial_guess, plot=True, seed=None, tune=500):
    """
    Uses pymc3 to calculate the trace for the PSP model. In this particular model, we assume a single PSP peak and
    a constant baseline.
    
    Parameters
    ----------
    data : NeuronalData
        Imported data
    nsamples : integer
        Number of samples for pymc3 calculation
    initial_guess : dictionary
        Dictionary of initial guesses for the parameters of the model
    plot : boolean
        Plots of the marginal distributions of the estimated parameters (plotted when True)
    seed : int or list of ints, optional
        Random seed for pymc3 sampling, defaults to None
    tune : int
        Number of iterations to tune in pymc3 sampling, defaults to 500

    Returns
    -------
    trace : pymc3.backends.base.MultiTrace
        A MultiTrace object containing the samples
    """
    
    with pm.Model() as PSP_model:
        num_psp = data.num_psp
        t = data.data['T']
        # Set parameter ranges
        b = pm.Flat('b')
        sigma = pm.HalfFlat('sigma')
        a = [pm.Flat('a' + str(i)) for i in range(num_psp)]
        t_psp = [pm.Uniform('t_psp' + str(i), lower=np.min(t), upper=np.max(t)) for i in range(num_psp)]
        tau_d = [pm.Uniform('tau_d' + str(i), lower=0, upper=0.1) for i in range(num_psp)]
        tau_r = [pm.Uniform('tau_r' + str(i), lower=0, upper=0.1) for i in range(num_psp)]

        log_likelihood = psp_log_likelihood(data, b, sigma, a, t_psp, tau_d, tau_r)
        pm.Potential('result', log_likelihood)
        trace=pm.sample(nsamples, cores=2, start=initial_guess, random_seed=seed, tune=tune)

    if plot:
        pm.traceplot(trace)
    return trace
