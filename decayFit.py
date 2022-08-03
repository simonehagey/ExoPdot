import utils
import corner
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def model(s, epochs):
    """
    Defines quadratic transit timing model (for orbital decay)
    """
    t0, P0, PdE = s
    epochs = np.array(epochs, dtype=np.float64)
    tcs = t0 + P0 * epochs + (1/2)*(epochs**2)*PdE
    return tcs

def draw(limits, s, i, widths):
    """
    Draw variables for a new trial (uniform priors)
    """
    new_state = s.copy()
    v = np.random.normal(s[i], widths[i])        # draw new variable
    while v < limits[i][0] or v > limits[i][1]:  # check if within limits
        v = np.random.normal(s[i], widths[i])
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
        chi2_0 = chi2 * 1
        alpha = 1.0
    elif chi2_0 > chi2:
        alpha = 1.0
    else:
        alpha = np.exp((chi2_0 - chi2))

    return alpha, chi2_0, chi2

def main(data, initial_state, burn_in, limits, niter, variables, widths, directory):
    """
    Main MCMC run - returns chain after burn-in
    """
    accepted = 0
    chi20 = 0
    var_accepted = [0,0,0]
    current_state = initial_state

    chain = np.empty((niter, len(variables)), dtype=np.float64)
    for i in tqdm(range(niter)):
        for var in variables:
            c = current_state.copy()
            proposal_state = draw(limits, c, var, widths)
            alpha, chi2_old, chi2_new = evaluate(data, proposal_state, i, chi20)

            if utils.random_coin(alpha):
                chain[i][var] = proposal_state[var]
                current_state = proposal_state.copy()
                chi20 = chi2_new
                x = var_accepted[var]
                var_accepted[var] = x + 1
                accepted += 1
            else:
                chain[i][var] = current_state[var]
                chi20 = chi2_old

    var_accepted = np.array(var_accepted)/niter
    burned_chain = chain[burn_in:]
    np.savetxt(directory + "_decay_burnedchain.txt", burned_chain)
    # np.savetxt(directory + "_decay_fullchain.txt", chain)
    # np.savetxt(directory + "_decay_accpt_ratios.txt", var_accepted)

    return burned_chain

def confidence(full_chain):
    """
    Calculates 68% confidence interval for orbital decay model parameters
    """
    q_t0 = corner.quantile(full_chain[:, 0], [0.16, 0.5, 0.84])
    q_P = corner.quantile(full_chain[:, 1], [0.16, 0.5, 0.84])
    q_PdE = corner.quantile(full_chain[:, 2], [0.16, 0.5, 0.84])
    vals = np.array([q_t0[1], q_P[1], q_PdE[1]])

    lower = np.array([q_t0[0], q_P[0], q_PdE[0]])
    lower = vals - lower
    upper = np.array([q_t0[2], q_P[2], q_PdE[2]])
    upper = upper - vals

    print("t0: ", str(vals[0]), "+", str(upper[0]), "-", str(lower[0]))
    print("P: ", str(vals[1]), "+", str(upper[1]), "-", str(lower[1]))
    print("PdE: ", str(vals[2]), "+", str(upper[2]), "-", str(lower[2]))
    conv = (365.25 * 24. * 3600. * 1e3) / vals[1]
    print("PdT: ", str(vals[2] * conv), "+", str(upper[2] * conv), "-", str(lower[2] * conv))

    return vals, upper, lower

def plots(CHAIN, directory):

    labels=["t0","P0","PdE"]
    conv = (365.25 * 24. * 3600. * 1e3) / np.mean(CHAIN[:, 1])

    fig, ax = plt.subplots(4, sharex=True)
    for i in range(len(labels)):
        ax[i].plot(CHAIN[:, i])
        ax[i].set_ylabel(labels[i])

    ax[3].plot(CHAIN[:, i]*conv)
    ax[3].set_ylabel("PdT")
    plt.xlabel("iteration")
    plt.savefig(directory + "_decay_trace")
    plt.close()

    corner.corner(CHAIN[::100,:], labels=labels, quantiles=[0.16, 0.5, 0.84],
                  show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(directory+"_decay_corner")
    plt.close()
