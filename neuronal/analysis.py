"""
Useful functions for analysing pymc3 traces.

Authors: Amelia Paine, Han Sae Jung
"""
import numpy as np


def get_quantiles(df):
    """
    Returns a DataFrame containing quantiles for each parameter

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the results from pymc3 calculation

    Returns
    -------
    pandas.DataFrame
        DataFrame containing quantiles
    """
    return df.quantile([0.16,0.50,0.84], axis=0)


def get_params_from_df(data, df):
    """
        Returns median (50th percentile) model parameters, grouped into numpy arrays when relevant

        Parameters
        ----------
        data : NeuronalData
            Imported data
        df : pandas.DataFrame
            Dataframe containing the results from pymc3 calculation

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
    quantile_df = get_quantiles(df)
    medians = quantile_df.loc[0.50]
    num_psp = data.num_psp
    b_start = medians['b_start']
    b_end = medians['b_end']
    sigma = medians['sigma']
    b = [medians['b__' + str(i)] for i in range(num_psp)]
    a = [medians['a__' + str(i)] for i in range(num_psp)]
    t_psp = [medians['t_psp__' + str(i)] for i in range(num_psp)]
    tau_d = [medians['tau_d__' + str(i)] for i in range(num_psp)]
    tau_r = [medians['tau_r__' + str(i)] for i in range(num_psp)]
    return sigma, b_start, b_end, b, a, t_psp, tau_d, tau_r


def calculate_peak_amplitudes(i, df, t_end, num_psp):
    """
    Adds a peak amplitude column for the i-th PSP to the pandas.DataFrame containing the pymc3 calculation results

    Parameters
    ----------
    i : integer
        Label for a PSP peak
    df : pandas.DataFrame
        DataFrame containing the pymc3 trace
    t_end : float
        End time of the data
    num_psp : integer
        Number of PSP's

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with an additional peak amplitude column for the i-th PSP
    """
    num_sample = df.shape[0]
    a = df.loc[:, ['a__' + str(i)]].values
    tau_d = df.loc[:, ['tau_d__' + str(i)]].values
    tau_r = df.loc[:, ['tau_r__' + str(i)]].values
    t_psp = df.loc[:, ['t_psp__' + str(i)]].values
    b = df.loc[:, ['b__' + str(i)]].values
    if i < num_psp-1:
        b_next = df.loc[:, ['b__' + str(i+1)]].values
        t_psp_next = df.loc[:, ['t_psp__' + str(i+1)]].values
    elif i == num_psp-1:
        b_next = df.loc[:, ['b_end']].values
        t_psp_next = t_end
    # t_max is defined with respect to t_psp (t_psp = 0 with respect to t_max)
    t_max = np.log(tau_d / tau_r) / (1/tau_r - 1/tau_d)
    df['peak_amp__' + str(i)] = a * (np.exp(-t_max / tau_d) - np.exp(-t_max / tau_r)) -\
                                (b_next - b) / (t_psp_next - t_psp) * t_max
    return df


def report_best_fit(quantile_df, parameter, print_fit=True):
    """
    Adds a peak amplitude column for the i-th PSP to the pandas.DataFrame containing the pymc3 calculation results

    Parameters
    ----------
    i : integer
        Label for a PSP peak
    quantile_df : pandas.DataFrame
        DataFrame containing the 16th, 50th, and 84th percentile results for the parameter
    parameter : string
        Name of the parameter

    Returns
    -------
    value_minus : float
        Parameter's value at 16th percentile (about 1 standard deviation below the mean if we assume a Gaussian distribution)
    value_mid : float
        Parameter's value at 50th percentile (median of the marginal distribution)
    value_plus : float
        Parameter's value at 84th percentile (about 1 standard deviation above the mean if we assume a Gaussian distribution)
    """
    value_mid = quantile_df.loc[0.50, parameter]
    value_plus = quantile_df.loc[0.84, parameter]
    value_minus = quantile_df.loc[0.16, parameter]
    if print_fit:
        print("The best fit value for {} is {:.4f} + {:.4f} - {:.4f}".format(parameter, value_mid,
              value_plus - value_mid, value_mid - value_minus))
    return value_minus, value_mid, value_plus