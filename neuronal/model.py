import theano.tensor as tt
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def psp_log_likelihood(data, theta):
    """
    Calculates log likelihood for a single psp assuming constant baseline
    
    Parameters
    ----------
    data : NeuronalData
        Imported data
    theta : list
        List of the parameters of the model
    """
    b, sigma, a1, t1, tau_d1, tau_r1 = theta
    t = np.array(data.data['T'])
    v = np.array(data.data['V'])
            
    single_psp_model = (t >= t1) * a1 * (np.exp(-(t-t1) / tau_d1) - np.exp(-(t-t1) / tau_r1)) + b
        
    residual = ((v - single_psp_model) / sigma)**2
    constant = 1 / np.sqrt(2*np.pi*sigma**2)
    log_likelihood = (np.log(constant) - 0.5 * residual).sum()
    if not np.isfinite(log_likelihood):
        return -np.inf
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
    """
    
    with pm.Model() as PSP_model:
        t = np.array(data.data['T'])
        v = np.array(data.data['V'])
        
        b = pm.Flat('b')
        sigma = pm.HalfFlat('sigma')
        a1 = pm.Flat('a1')
        t1 = pm.Uniform('t1', lower=np.min(t), upper=np.max(t))
        tau_d1 = pm.Uniform('tau_d1', lower=0, upper=0.1)
        tau_r1 = pm.Uniform('tau_r1', lower=0, upper=0.1)
        
        model = (t >= t1) * a1 * (tt.exp(-(t-t1) / tau_d1) - tt.exp(-(t-t1) / tau_r1)) + b
        loglike = pm.Normal.dist(mu=model, sd=sigma).logp(v)
        pm.Potential('result', loglike)
        
        trace=pm.sample(nsamples, cores=2, start=initial_guess, random_seed=seed, tune=tune)

    if plot:
        pm.traceplot(trace)
    return trace

def psp3_fit(data, nsamples, initial_guess, plot=True, seed=None, tune=500):
    """
    Uses pymc3 to calculate the trace for the PSP model. In this particular model, we assume three PSP peaks and
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
    """
    with pm.Model() as PSP_model:
        t = np.array(data.data['T'])
        v = np.array(data.data['V'])
        
        b = pm.Flat('b')
        sigma = pm.HalfFlat('sigma')
        a1 = pm.Flat('a1')
        t1 = pm.Uniform('t1', lower=np.min(t), upper=np.max(t))
        tau_d1 = pm.Uniform('tau_d1', lower=0, upper=0.1)
        tau_r1 = pm.Uniform('tau_r1', lower=0, upper=0.1)
        a2 = pm.Flat('a2')
        t2 = pm.Uniform('t2', lower=np.min(t), upper=np.max(t))
        tau_d2 = pm.Uniform('tau_d2', lower=0, upper=0.1)
        tau_r2 = pm.Uniform('tau_r2', lower=0, upper=0.1)
        a3 = pm.Flat('a3')
        t3 = pm.Uniform('t3', lower=np.min(t), upper=np.max(t))
        tau_d3 = pm.Uniform('tau_d3', lower=0, upper=0.1)
        tau_r3 = pm.Uniform('tau_r3', lower=0, upper=0.1)
        
        model = (t >= t1) * a1 * (tt.exp(-(t-t1) / tau_d1) - tt.exp(-(t-t1) / tau_r1)) + b +\
                (t >= t2) * a2 * (tt.exp(-(t-t2) / tau_d2) - tt.exp(-(t-t2) / tau_r2)) +\
                (t >= t3) * a3 * (tt.exp(-(t-t3) / tau_d3) - tt.exp(-(t-t3) / tau_r3))
        loglike = pm.Normal.dist(mu=model, sd=sigma).logp(v)
        pm.Potential('result', loglike)
        
        trace=pm.sample(nsamples, cores=2, start=initial_guess, random_seed=seed, tune=tune)

    if plot:
        pm.traceplot(trace)
    return trace
