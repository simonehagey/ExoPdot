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
    tcs = []
    for E in epochs:
        tcs.append(t0 + P0 * E + (1 / 2) * (E ** 2) * PdE)
    return np.array(tcs)

def draw(limits, s, variable_ind, widths):
    """
    Draw variables for a new trial (uniform priors)
    """
    new_state = s.copy()
    for i in variable_ind:
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
    chi2 = np.sum([(M - D) ** 2/err**2 for M, D, err in zip(m, tc, err)])

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
    var_accepted = [0,0,0]

    chain = []
    x2s = []
    current_state = initial_state
    chi20 = 0
    iteration = 0
    for i in tqdm(range(niter)):
        x2s.append(chi20)

        for var in variables:
            c = current_state.copy()
            proposal_state = draw(limits, c, [var], widths)
            alpha, chi2_old, chi2_new = evaluate(data, proposal_state, iteration, chi20)

            if utils.random_coin(alpha):
                chain.append(proposal_state)
                current_state = proposal_state.copy()
                chi20 = chi2_new
                x = var_accepted[var]
                var_accepted[var] = x + 1
                accepted += 1
            else:
                chain.append(current_state)
                chi20 = chi2_old
            iteration +=1

    # print("total acceptance ratio:", accepted/(len(variables)*niter))
    # print("t0 acceptance ratio:", var_accepted[0] / niter)
    # print("P0 acceptance ratio:", var_accepted[1] / niter)
    # print("PdE acceptance ratio:", var_accepted[2] / niter)

    chain = np.array(chain)
    thinned_chain = chain[::len(variables)]
    burned_chain = thinned_chain[burn_in:]

    np.savetxt(directory+"_decay_burnedchain.txt", burned_chain)

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

    plt.plot(CHAIN[:, 0])
    plt.title("t0")
    plt.savefig(directory+"_decay_t0")
    plt.close()
    plt.close()

    plt.plot(CHAIN[:, 1])
    plt.title("P0")
    plt.savefig(directory+"_decay_P0")
    plt.close()

    plt.plot(CHAIN[:, 2])
    plt.title("PdE")
    plt.savefig(directory+"_decay_PdE")
    plt.close()

    corner.corner(CHAIN, labels=["t0","P0","PdE"], quantiles=[0.16, 0.5, 0.84],
                  show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(directory+"_decay_corner")
    plt.close()
