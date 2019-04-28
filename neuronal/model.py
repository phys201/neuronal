"""
Uses a generative model to fit parameters to neuronal data.

Authors: Amelia Paine, Han Sae Jung
"""

import theano.tensor as tt
import pymc3 as pm
import numpy as np


#def psp_log_likelihood(data, b, sigma, a, t_psp, tau_d, tau_r):
#    """
#    Calculates log likelihood for a single psp assuming constant baseline
#    
#    Parameters
#    ----------
#    data : NeuronalData
#        Experimental data
#    b : float
#        Constant baseline value
#    sigma : float
#        Width of Gaussian noise
#    a : list of float
#        amplitudes of PSPs
#    t_psp : list of float
#        PSP start times
#    tau_d : list of float
#        Decay constants (PSP long-range behavior)
#    tau_r : lost of float
#        Rise constants (PSP short-range behavior)
#
#    Returns
#    -------
#    log_likelihood : TensorVariable
#        Log-likelihood of parameters
#    """
#    num_psp = data.num_psp
#
#    if any(len(x) != num_psp for x in [a, t_psp, tau_d, tau_r]):
#        raise ValueError('Number of parameters is inconsistent with data.num_psp. Make sure num_psp is set '
#                         'and that a, t_psp, tau_d, and tau_r are the correct length.')
#
#    t = np.array(data.data['T'])
#    v = np.array(data.data['V'])
#
#    # TODO: broadcasting?
#    model = b + np.sum(
#        [(t >= t_psp[i]) * a[i] * (tt.exp(-(t - t_psp[i]) / tau_d[i]) - tt.exp(-(t - t_psp[i]) / tau_r[i]))
#         for i in range(num_psp)])
#    residual = ((v - model) / sigma) ** 2
#    constant = 1 / np.sqrt(2 * np.pi * sigma ** 2)
#    log_likelihood = (np.log(constant) - 0.5 * residual).sum()
#    return log_likelihood


#def psp_fit(data, nsamples, initial_guess, plot=True, seed=None, tune=500):
#    """
#    Uses pymc3 to calculate the trace for the PSP model. In this particular model, we assume a single PSP peak and
#    a constant baseline.
#    
#    Parameters
#    ----------
#    data : NeuronalData
#        Imported data
#    nsamples : int
#        Number of samples for pymc3 calculation
#    initial_guess : dict
#        Dictionary of initial guesses for the parameters of the model
#    plot : boolean
#        Plots of the marginal distributions of the estimated parameters (plotted when True)
#    seed : int or list of int, optional
#        Random seed for pymc3 sampling, defaults to None
#    tune : int
#        Number of iterations to tune in pymc3 sampling, defaults to 500
#
#    Returns
#    -------
#    trace : pymc3.backends.base.MultiTrace
#        A MultiTrace object containing the samples
#    """
#    
#    with pm.Model() as PSP_model:
#        num_psp = data.num_psp
#        t = data.data['T']
#        # Set parameter ranges
#        b = pm.Flat('b')
#        sigma = pm.HalfFlat('sigma')
#        a = [pm.Flat('a' + str(i)) for i in range(num_psp)]
#        t_psp = [pm.Uniform('t_psp' + str(i), lower=np.min(t), upper=np.max(t)) for i in range(num_psp)]
#        tau_d = [pm.Uniform('tau_d' + str(i), lower=0, upper=0.1) for i in range(num_psp)]
#        tau_r = [pm.Uniform('tau_r' + str(i), lower=0, upper=0.1) for i in range(num_psp)]
#
#        log_likelihood = psp_log_likelihood(data, b, sigma, a, t_psp, tau_d, tau_r)
#        pm.Potential('result', log_likelihood)
#        trace=pm.sample(nsamples, cores=2, start=initial_guess, random_seed=seed, tune=tune)
#
#    if plot:
#        pm.traceplot(trace)
#    return trace

def psp_log_likelihood(data, b_start, b, b_end, sigma, a, t_psp, tau_d, tau_r):
    """
    Calculates log likelihood for psp's with piecewise, linearly varying baseline
    
    Parameters
    ----------
    data : NeuronalData
        Experimental data
    b_start : float
        Constant baseline before the first PSP
    b : list float
        Constant baselines
    b_end : float
        Constant baseline after the last PSP
    sigma : float
        Width of Gaussian noise 
    a : list of float
        Constant related to the amplitudes of PSPs
    t_psp : list of float
        PSP start times
    tau_d : list of float
        Decay constants (PSP long-range behavior)
    tau_r : list of float
        Rise constants (PSP short-range behavior)

    Returns
    -------
    log_likelihood : float
        Log-likelihood of parameters
    """
    num_psp = data.num_psp

    if any(len(x) != num_psp for x in [a, t_psp, tau_d, tau_r]):
        raise ValueError('Number of parameters is inconsistent with data.num_psp. Make sure num_psp is set '
                         'and that a, t_psp, tau_d, and tau_r are the correct length.')

    t = np.array(data.data['T'])
    v = np.array(data.data['V'])
    
    #needs to be vectorized
    model = (t <= t_psp[0]) * (b_start + (b[0] - b_start) / (t_psp[0] - t[0]) * (t - t[0])) +\
            np.sum([
                    (t >= t_psp[i]) * (a[i] * (np.exp(-(t-t_psp[i]) / tau_d[i]) - np.exp(-(t-t_psp[i]) / tau_r[i])) +\
                    (t <= t_psp[i+1]) * (b[i] + (b[i+1] - b[i]) / (t_psp[i+1] - t_psp[i]) * (t - t_psp[i])))
                    for i in range(num_psp - 1)], axis=0) +\
            (t >= t_psp[-1]) * (a[-1] * (np.exp(-(t-t_psp[-1]) / tau_d[-1]) - np.exp(-(t-t_psp[-1]) / tau_r[-1])) +\
            (b[-1] + (b_end - b[-1]) / (t[-1] - t_psp[-1]) * (t - t_psp[-1])))
    
    residual = ((v - model) / sigma) ** 2
    constant = 1 / np.sqrt(2 * np.pi * sigma ** 2)
    log_likelihood = (np.log(constant) - 0.5 * residual).sum()
    return log_likelihood

def psp_fit(data, nsamples, initial_guess, plot=True, seed=None, tune=500):
    """
    Uses pymc3 to calculate the trace for the PSP model. In this particular model, we assume piecewise, linearly varying
    baselines
    
    Parameters
    ----------
    data : NeuronalData
        Imported data
    nsamples : int
        Number of samples for pymc3 calculation
    initial_guess : dict
        Dictionary of initial guesses for the parameters of the model
    plot : boolean
        Plots of the marginal distributions of the estimated parameters (plotted when True)
    seed : int or list of int, optional
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
        b_start = pm.Flat('b_start')
        b = pm.Flat('b', shape=num_psp)
        b_end = pm.Flat('b_end')
        sigma = pm.HalfFlat('sigma')
        a = pm.Flat('a', shape=num_psp)
        t_psp = pm.Uniform('t_psp', lower=np.min(t), upper=np.max(t), shape=num_psp)
        tau_d = pm.Uniform('tau_d', lower=0, upper=0.1, shape=num_psp)
        tau_r = pm.Uniform('tau_r', lower=0, upper=0.1, shape=num_psp)
        
        log_likelihood = psp_log_likelihood(data, b_start, b, b_end, sigma, a, t_psp, tau_d, tau_r)
        pm.Potential('result', log_likelihood)
        trace=pm.sample(nsamples, cores=2, start=initial_guess, random_seed=seed, tune=tune)

    if plot:
        pm.traceplot(trace)
    return trace
