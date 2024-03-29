import utils
import corner
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from numpy import random
import numpy.random as rand
import csv

def model(s, epochs):
    """
    Defines apsidal precession transit timing model
    """
    t0, Ps, e, w0, wdE = s
    E = np.array(epochs, dtype=np.float64)
    tcs = t0 + Ps*E - (e*(Ps / (1-wdE/(2*np.pi))) / np.pi) * np.cos(w0 + E * wdE)
    return np.array(tcs)

def draw(limits, s, i, widths, linear_results, linear_sig):
    """
    Draw variables for a new trial, priors are:
        t0 (reference transit time) - uniform prior
        P_s (sidereal period) - normal dist. prior centered on the best-fit result from the constant period fit
        w0 (phase) - uniform prior
        e (eccentricity) - log-uniform prior
        wdE (precession rate /epoch) - log-uniform prior
    """
    new_state = s.copy()
    if i == 0:  # t0
        v = np.random.normal(s[i], widths[i])
        while v < limits[i][0] or v > limits[i][1]:
            v = np.random.normal(s[i], widths[i])

    if i == 1:  # P_s
        means = np.array([s[i],linear_results[1]])
        cov = [[widths[i]**2,0],
               [0,linear_sig[1]**2]]
        v = rand.multivariate_normal(means, cov)[1]
        while v < limits[i][0] or v > limits[i][1]:
            v = rand.multivariate_normal(means, cov)[1]

    if i == 2:  # e
        p = np.random.normal(np.log10(s[i]), 0.03)
        v = 10**p
        while v < limits[i][0] or v > limits[i][1]:
            p = np.random.normal(np.log10(s[i]), 0.03)
            v = 10 ** p

    if i == 3:  # w0
        v = np.random.normal(s[i], widths[i])
        v = utils.wrap(v)
        while v < limits[i][0] or v > limits[i][1]:
            v = np.random.normal(s[i], widths[i])
            v = utils.wrap(v)

    if i == 4:  # wdE
        p = np.random.normal(np.log10(s[i]), 0.03)
        v = 10 ** p
        while v < limits[i][0] or v > limits[i][1]:
            p = np.random.normal(np.log10(s[i]), 0.03)
            v = 10 ** p

    new_state[i] = v
    return new_state

def evaluate(data, pstate, iter, chi2_0):
    """
    Evaluates a new parameter set
    """
    epochs, tc, err = data

    m = model(pstate, epochs)
    tc = np.array(tc, dtype=np.float64)
    err = np.array(err, dtype=np.float64)
    chi2 = np.sum((m - tc)**2/err**2)

    if iter == 0:
        chi2_0 = chi2*1
        alpha = 1.0
    elif chi2_0 > chi2:
        alpha = 1.0
    else:
        alpha = np.exp(chi2_0-chi2)

    return alpha, chi2_0, chi2

def main(data, initial_state, burn_in, limits, niter, variables, widths, linear_results, linear_sig, directory):
    """
    Main MCMC run - returns chain after burn-in
    """
    accepted = 0
    chi20 = 0
    var_accepted = [0,0,0,0,0]
    current_state = initial_state

    chain = np.empty((niter, len(variables)), dtype=np.float64)
    for i in tqdm(range(niter)):
        for var in variables:
            c = current_state.copy()
            proposal_state = draw(limits, c, var, widths, linear_results, linear_sig)
            alpha, chi2_old, chi2_new = evaluate(data, proposal_state, i, chi20)

            if utils.random_coin(alpha):
                chain[i][var] = proposal_state[var]
                current_state = proposal_state.copy()
                chi20 = chi2_new
                var_accepted[var] += 1
                accepted += 1
            else:
                chain[i][var] = current_state[var]
                chi20 = chi2_old

    var_accepted = np.array(var_accepted)/niter
    burned_chain = chain[burn_in:]
    np.savetxt(directory+"_precession_burnedchain.txt", burned_chain)
    # np.savetxt(directory+"_precession_fullchain.txt", chain)
    # np.savetxt(directory + "_precession_accpt_ratios.txt", var_accepted)

    return burned_chain

def confidence(full_chain):
    """
    Calculates 68% confidence interval for apsidal precession model parameters
    """
    q_t0 = corner.quantile(full_chain[:,0],[0.16,0.5,0.84])
    q_P = corner.quantile(full_chain[:,1],[0.16,0.5,0.84])
    q_e = corner.quantile(full_chain[:,2],[0.16,0.5,0.84])
    q_w0 = corner.quantile(full_chain[:,3],[0.16,0.5,0.84])
    q_wdE = corner.quantile(full_chain[:,4],[0.16,0.5,0.84])

    vals = np.array([q_t0[1],q_P[1],q_e[1],q_w0[1],q_wdE[1]])
    lower = np.array([q_t0[0],q_P[0],q_e[0],q_w0[0],q_wdE[0]])
    lower = vals - lower
    upper = np.array([q_t0[2],q_P[2],q_e[2],q_w0[2],q_wdE[2]])
    upper = upper-vals

    print("t0: ",str(vals[0]),"+",str(upper[0]),"-",str(lower[0]))
    print("P: ", str(vals[1]), "+", str(upper[1]), "-", str(lower[1]))
    print("e: ", str(vals[2]), "+", str(upper[2]), "-", str(lower[2]))
    print("w0: ", str(vals[3]), "+", str(upper[3]), "-", str(lower[3]))
    print("wdE: ", str(vals[4]), "+", str(upper[4]), "-", str(lower[4]))
    return vals, upper, lower

def plots(CHAIN, directory):

    labels = ["t0", "Ps", "e", "w0", "wdE"]

    fig, ax = plt.subplots(len(labels), sharex=True)
    for i in range(len(ax)):
        ax[i].plot(CHAIN[:, i])
        ax[i].set_ylabel(labels[i])
    plt.xlabel("iteration")
    plt.savefig(directory+"_precession_trace")
    plt.close()

    corner.corner(CHAIN[::1000,:], labels=labels, quantiles=[0.16, 0.5, 0.84],
                  show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(directory+"_precession_corner")
    plt.close()