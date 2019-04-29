"""
Uses a generative model to fit parameters to neuronal data.

Authors: Amelia Paine, Han Sae Jung
"""

import pymc3 as pm
import numpy as np
import warnings


def psp_model(data, b_start, b, b_end, a, t_psp, tau_d, tau_r):
    """
    Calculates a PSP model with piecewise, linearly varying baselines
    
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
    model : Numpy Array
        PSP model of the given parameters
    """
    num_psp = data.num_psp
    t = np.array(data.data['T'])
    
    #needs to be vectorized
    model = (t <= t_psp[0]) * (b_start + (b[0] - b_start) / (t_psp[0] - t[0]) * (t - t[0])) +\
            np.sum([
                    (t >= t_psp[i]) * (a[i] * (np.exp(-(t-t_psp[i]) / tau_d[i]) - np.exp(-(t-t_psp[i]) / tau_r[i])) +\
                    (t <= t_psp[i+1]) * (b[i] + (b[i+1] - b[i]) / (t_psp[i+1] - t_psp[i]) * (t - t_psp[i])))
                    for i in range(num_psp - 1)], axis=0) +\
            (t >= t_psp[-1]) * (a[-1] * (np.exp(-(t-t_psp[-1]) / tau_d[-1]) - np.exp(-(t-t_psp[-1]) / tau_r[-1])) +\
            (b[-1] + (b_end - b[-1]) / (t[-1] - t_psp[-1]) * (t - t_psp[-1])))
    return model


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
    v = np.array(data.data['V'])

    model = psp_model(data, b_start, b, b_end, a, t_psp, tau_d, tau_r)

    residual = ((v - model) / sigma) ** 2
    constant = 1 / np.sqrt(2 * np.pi * sigma ** 2)
    log_likelihood = (np.log(constant) - 0.5 * residual).sum()
    return log_likelihood


def psp_fit(data, nsamples, initial_guess, plot=True, seed=None, tune=500, suppress_warnings=False):
    """
    Uses pymc3 to calculate the trace for the PSP model. We assume a piecewise, linearly varying baseline.

    Keys for initial_guess (for n PSPs)
    -----------------------------------
    b_start, b_end : start and end baseline values
    b : (list) baseline values at each PSP start
    sigma : width of Gaussian noise
    a : (list) amplitude of each PSP
    t_psp : (list) start time for each PSP
    tau_d : (list) decay time constant of each PSP
    tau_r : (list) rise time constant of each PSP
    
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
    tune : int, optional
        Number of iterations to tune in pymc3 sampling, defaults to 500
    suppress_warnings : bool, optional
        Hide warnings if initial guess doesn't look right

    Returns
    -------
    trace : pymc3.backends.base.MultiTrace
        A MultiTrace object containing the samples
    """
    if not suppress_warnings:
        validate_params(data, initial_guess)

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


def validate_params(data, params):
    """
    Checks if a certain dictionary of parameters is appropriate for describing the given data

    Parameters
    ----------
    data : NeuronalData
        Imported data
    params : dict
        Dictionary of parameters of the model
    """
    num_psp = data.num_psp
    keys = params.keys()
    expected_keys = ['b_start', 'b_end', 'b', 'sigma', 'a', 't_psp', 'tau_d', 'tau_r']
    for key in expected_keys:
        if key not in keys:
            warnings.warn('Missing parameter ' + key)
    for key in keys:
        if key not in expected_keys:
            warnings.warn('Unexpected parameter ' + key)
    for key in ['b', 'a', 't_psp', 'tau_d', 'tau_r']:
        if key in keys:
            if not isinstance(params[key], list):
                warnings.warn('Incorrect parameter type: ' + key + ' is ' + str(type(params[key])) + ', should be list')
            elif len(params[key]) != num_psp:
                warnings.warn('Length of list ' + key + ' inconsistent with num_psp')

