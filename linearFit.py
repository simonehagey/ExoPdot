import utils
import corner
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def model(s, epochs):
    """
    Defines constant period transit timing model
    """
    t0, P0 = s
    epochs = np.array(epochs, dtype=np.float64)
    tcs = t0 + P0 * epochs
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
    var_accepted = [0,0]
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
    np.savetxt(directory + "_linear_burnedchain.txt", burned_chain)
    np.savetxt(directory + "_linear_fullchain.txt", chain)
    np.savetxt(directory + "_linear_accpt_ratios.txt", var_accepted)

    return burned_chain

def confidence(full_chain):
    """
    Calculates 68% confidence interval for constant period model parameters
    """
    q_t0 = corner.quantile(full_chain[:, 0], [0.16, 0.5, 0.84])
    q_P = corner.quantile(full_chain[:, 1], [0.16, 0.5, 0.84])
    vals_l = np.array([q_t0[1], q_P[1]])

    lower_l = np.array([q_t0[0], q_P[0]])
    lower_l = vals_l - lower_l
    upper_l = np.array([q_t0[2], q_P[2]])
    upper_l = upper_l - vals_l

    print("t0: ", str(vals_l[0]), "+", str(upper_l[0]), "-", str(lower_l[0]))
    print("P: ", str(vals_l[1]), "+", str(upper_l[1]), "-", str(lower_l[1]))

    return vals_l, upper_l, lower_l

def plots(CHAIN, directory):

    labels=["t0","P0"]

    fig, ax = plt.subplots(len(labels), sharex=True)
    for i in range(len(ax)):
        ax[i].plot(CHAIN[:, i])
        ax[i].set_ylabel(labels[i])
    plt.xlabel("iteration")
    plt.savefig(directory + "_linear_trace")
    plt.close()

    corner.corner(CHAIN[::100,:], labels=labels, quantiles=[0.16, 0.5, 0.84],
                  show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(directory+"_linear_corner")
    plt.close()
