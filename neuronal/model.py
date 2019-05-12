"""
Uses a generative model to fit parameters to neuronal data.

Authors: Amelia Paine, Han Sae Jung
"""

from .io import NeuronalData
import pymc3 as pm
import numpy as np
import pandas as pd
import theano
import theano.tensor as tt
import warnings


def psp_model(data, b, a, t_psp, tau_d, tau_r, start_index=0, end_index=-1):
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
    model : theano something
        PSP model of the given parameters
    """
    eps = 1e-5
    num_psp = data.num_psp
    t = np.array(data.data['T'][start_index:end_index])

    # needs to be vectorized
    t_resize = theano.shared(np.tile(t, (num_psp + 2, 1)).astype("float64"))
    t_psp_resize = tt.tile(t_psp, (len(t), 1)).T

    t_psp_next = t_psp + tt.concatenate([tt.extra_ops.diff(t_psp), theano.shared(np.array([eps]).astype("float64"))],
                                        axis=0)
    t_psp_next_resize = tt.tile(t_psp_next, (len(t), 1)).T

    a_resize = tt.tile(a, (len(t), 1)).T
    b_resize = tt.tile(b, (len(t), 1)).T
    tau_d_resize = tt.tile(tau_d, (len(t), 1)).T
    tau_r_resize = tt.tile(tau_r, (len(t), 1)).T

    b_next = b + tt.concatenate([tt.extra_ops.diff(b), theano.shared(np.array([eps]).astype("float64"))], axis=0)
    b_next_resize = tt.tile(b_next, (len(t), 1)).T

    model = tt.sum((t_resize >= t_psp_resize) * (a_resize * (
                tt.exp((t_resize >= t_psp_resize) * -(t_resize - t_psp_resize) / tau_d_resize) - tt.exp(
            (t_resize >= t_psp_resize) * -(t_resize - t_psp_resize) / tau_r_resize)) +
                                                 (t_resize < t_psp_next_resize) * (
                                                             b_resize + (b_next_resize - b_resize) / (
                                                                 t_psp_next_resize - t_psp_resize) * (
                                                                         t_resize - t_psp_resize))), axis=0)
    return model


def simulate_psp_data(sigma, b_start, b, b_end, a, t_psp, tau_d, tau_r, t_range, step):
    """
        Simulates experimental data with the given parameters and Gaussian noise

        Parameters
        ----------
        sigma : float
            Width of Gaussian noise
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
        t_range : tuple
            Desired time range (start, end)
        step : float
            Desired time step

        Returns
        -------
        simulated_data : pandas.DataFrame
            DataFrame containing time ('T') vs. voltage ('V') simulated data
        """
    t_vals = np.arange(t_range[0], t_range[1], step)
    dummy_data = NeuronalData(pd.DataFrame({
        'T': t_vals,
        'V': t_vals
    }), num_psp=len(t_psp))
    model = psp_model(dummy_data, b_start, b, b_end, a, t_psp, tau_d, tau_r)
    noise = np.random.normal(0, sigma, len(model))
    simulated_data = NeuronalData(pd.DataFrame({
        'T': t_vals,
        'V': model + noise
    }), num_psp=len(t_psp))
    return simulated_data


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


def psp_fit(data, nsamples, initial_guess, plot=True, seed=None, tune=500, cores=1, suppress_warnings=False,
            prior_ranges=()):
    """
    Uses pymc3 to calculate the trace for the PSP model. We assume a piecewise, linearly varying baseline.

    Keys for initial_guess and prior_ranges (for n PSPs)
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
    cores : int, optional
        Number of cores for pymc3 parallelization
    suppress_warnings : bool, optional
        Hide warnings if initial guess doesn't look right
    prior_ranges : dict, optional
        Dictionary of tuples denoting uniform prior ranges (start, end) for each parameter

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing the trace
    """
    if not suppress_warnings:
        validate_params(data, initial_guess)

    with pm.Model() as PSP_model:
        num_psp = data.num_psp
        t = data.data['T']

        # Default priors
        default_priors = {
            'b_start': "pm.Flat('b_start')",
            'b_end': "pm.Flat('b_end')",
            'b': "pm.Flat('b', shape=num_psp)",
            'sigma': "pm.HalfFlat('sigma')",
            'a': "pm.Flat('a', shape=num_psp)",
            't_psp': "pm.Uniform('t_psp', lower=np.min(t), upper=np.max(t), shape=num_psp)",
            'tau_d': "pm.Uniform('tau_d', lower=0, upper=0.1, shape=num_psp)",
            'tau_r': "pm.Uniform('tau_r', lower=0, upper=0.1, shape=num_psp)"
        }
        # Update with user-specified priors
        priors = {}
        for param in default_priors:
            if param in prior_ranges:
                if not suppress_warnings and param not in default_priors:
                    warnings.warn('Unexpected prior specification: nonexistent parameter ' + str(param))
                elif param in ['b_start', 'b_end', 'sigma']:
                    priors[param] = pm.Uniform(param, lower=prior_ranges[param][0], upper=prior_ranges[param][1])
                else:
                    priors[param] = pm.Uniform(param, lower=prior_ranges[param][0],
                                               upper=prior_ranges[param][1], shape=num_psp)
            else:
                priors[param] = eval(default_priors[param])

        log_likelihood = psp_log_likelihood(data, priors['b_start'], priors['b'], priors['b_end'], priors['sigma'],
                                            priors['a'], priors['t_psp'], priors['tau_d'], priors['tau_r'])
        pm.Potential('result', log_likelihood)
        trace = pm.sample(nsamples, cores=cores, start=initial_guess, random_seed=seed, tune=tune)

    if plot:
        pm.traceplot(trace)
    df = pm.trace_to_dataframe(trace)
    return df


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

