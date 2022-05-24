import utils
import linearFit
import decayFit
import precessionFit
import numpy as np
from scipy.optimize import curve_fit

def fitAll(data, niter, burn_in, planetinfo, directory, fit_precess=False, diagnostic_plots=False):
    """
    Fits all three transit timing models: constant period, orbital decay, and apsidal precession.
    Fitting the precession model is optional as indicated by the fit_precess variable
    """
    # extract relevant planetary system info
    t0 = planetinfo["tc"][0]
    P0 = planetinfo["p"][0]
    target = planetinfo["target"]

    # extract data, number of MCMC iterations, and number of burn-in iterations
    epochs, observations, errs = data
    niter_L, niter_D, niter_P = niter
    burn_L, burn_D, burn_P = burn_in        # ToDo: include fitting eclipses
                                            # ToDo: combine 3 models into one MCMC

    """
    Fit linear model
    """
    # define parameters for MCMC
    variables_L = [0,1]                     # which variables to fit (t0-0, P0-1)
    widths_L = [0.0001, 0.0000001]          # step-widths
    limits_L = [(t0-1,t0+1),(P0-1,P0+1)]    # parameter limits
    init_L = [t0, P0]  # initial constant-period model parameters

    # fit constant period model
    print("Fitting constant period model (" + target + "):")
    chain_L = linearFit.main(data, init_L, burn_L, limits_L, niter_L, variables_L, widths_L, directory+target)
    results_L = linearFit.confidence(chain_L)
    linear = linearFit.model(results_L[0], epochs)
    if diagnostic_plots == True:
        linearFit.plots(chain_L, directory+target)

    """
    Fit decay model
    """
    # define parameters for MCMC
    variables_D = [0, 1, 2]    # which variables to fit (t0-0, P0-1, PdE-2)
    widths_D = [0.0001, 0.0000001, 1e-10]
    limits_D = [(results_L[0][0]-1,results_L[0][0]+1),(results_L[0][1]-1,results_L[0][1]+1),(-1e-7,1e-7)]
    init_D = [results_L[0][0], results_L[0][1], 0.0]

    # fit orbital decay model
    print("Fitting decay model ("+target+"):")
    chain_D = decayFit.main(data, init_D, burn_D, limits_D, niter_D, variables_D, widths_D, directory+target)
    results_D = decayFit.confidence(chain_D)
    decay = decayFit.model(results_D[0], epochs)
    if diagnostic_plots == True:
        decayFit.plots(chain_D, directory+target)

    """
    Fit precession model
    """
    # define these arrays of zeros to avoid issues if precession not fit
    results_P = [np.zeros(5),np.zeros(5),np.zeros(5)]
    init_P = np.zeros(5)

    # define parameters for MCMC
    variables_P = [0, 1, 2, 3, 4]                       # which variables to fit (t0-0, P0-1, e-2, w0-3, wdE-4)
    widths_P = [0.0001, 0.0000001, 0.0, 0.05, 0.0]      # step-widths
    limits_P = [(results_L[0][0] - 0.05, results_L[0][0] + 0.05),   # parameter limits
                (results_L[0][1] - 0.05, results_L[0][1] + 0.05),
                (0.00001, 0.1),                                     # 1e-5 < eccentricity < 1e-1
                (0, 2 * np.pi),                                     # 0 < phase < 2pi
                (0.000001, 0.001)]                                  # 1e-6 < precession rate < 1e-3

    # to aid convergence we pick initial precession model parameters from scipy.optimize.curve_fit
    # define precession model for scipy curve-fit functionality:
    if fit_precess == True:
        def function(epochs, t0, e, w0, wdE):
            tcs = []
            for E in range(len(epochs)):
                omega = w0 + epochs[E] * wdE
                Pa = results_L[0][1] / (1 - wdE / (2 * np.pi))
                tcs.append(t0 + results_L[0][1] * epochs[E] - (e * Pa / np.pi) * np.cos(omega))
            return np.array(tcs)

        # define limits a bit differently so the curve-fit explores more parameter space
        limits_curvefit = ([results_L[0][0] - 0.01, 0.00001, 0, 0.000001],
                           [results_L[0][0] + 0.01, 0.1, 2 * np.pi, 0.001])

        # use curve_fit to pick appropriate initial precession model parameters
        popt, pcov = curve_fit(function, epochs, observations, sigma=errs, bounds=limits_curvefit)
        init_P = [popt[0], results_L[0][1], popt[1], popt[2], popt[3]]

        # saving constant-period model fit and maxmimum period error for precession model MCMC draws
        linear_results = results_L[0].copy()
        linear_sig = np.array([max(results_L[1][0],results_L[2][0]),max(results_L[1][1],results_L[2][1])])

        # fit precession model
        print("Fitting apsidal precession model (" + target + "):")
        chain_P = precessionFit.main(data, init_P, burn_P, limits_P, niter_P,
                                     variables_P, widths_P, linear_results, linear_sig, directory+target)
        results_P = precessionFit.confidence(chain_P)
        if diagnostic_plots == True:
            precessionFit.plots(chain_P, directory+target)

    precession = precessionFit.model(results_P[0], epochs)  # outside of if-statement in case precession model not fit

    """
    DISPLAY AND SAVE RESULTS
    """
    if fit_precess == True:
        utils.printFit(directory+target,results_L,results_D,results_P,fit_precess)
    else:
        utils.printFit(directory+target,results_L,results_D)

    # compare models
    chisquare_linear, BIC_linear = utils.modelCompare(linear, observations, errs, variables_L)
    chisquare_decay, BIC_decay = utils.modelCompare(decay, observations, errs, variables_D)
    chisquare_precess, BIC_precess = utils.modelCompare(precession, observations, errs, variables_P)

    with open(directory + target + "_fit_results.txt", "a") as f:
        f.write("MODEL COMPARISON:\n")
        f.write("Linear model: "+"chi^2 = "+str(chisquare_linear)+"  BIC = "+str(BIC_linear)+"\n")
        f.write("Decay model: "+"chi^2 = "+str(chisquare_decay)+"  BIC = "+str(BIC_decay)+"\n")
        if fit_precess == True:
            f.write("Precession model: "+"chi^2 = "+str(chisquare_precess)+"  BIC = "+str(BIC_precess)+"\n")
    f.close()

    # save results
    data_fits = {}
    data_fits["DATA"] = [epochs,observations,errs]
    data_fits["MODELS"] = [linear,decay,precession]
    data_fits["PRIORS"] = [limits_L, limits_D, limits_P]
    data_fits["INITIAL"] = [init_L, init_D, init_P]
    data_fits["WIDTHS"] = [widths_L, widths_D, widths_P]
    data_fits["BIC"] = [BIC_linear,BIC_decay,BIC_precess]
    data_fits["CHISQUARE"] = [chisquare_linear, chisquare_decay, chisquare_precess]
    data_fits["LINEAR"] = results_L
    data_fits["DECAY"] = results_D
    data_fits["PRECESSION"] = results_P

    return data_fits

def iterate(data_fits, max_iters, niter, burn_in, planetinfo, save_directory):
    """
    Applies the iterative sigma-clipping method developed for data with high scatter
    """
    epochs, observations, errs = data_fits["DATA"]
    linear_model, decay_model, precession_model = data_fits["MODELS"]

    iters = 0
    epochs_removed = []
    obs_removed = []
    errs_removed = []
    for i in range(max_iters):

        # calculate residuals by removing best-fit orbital decay model
        residuals = np.array(observations) - np.array(decay_model)
        std = np.std(residuals)
        mean = np.mean(residuals)

        # flag data nominally outside of 3-sigma from the mean
        inds = []
        count = 0
        for i in range(len(residuals)):
            if (residuals[i]) < (mean - 3 * std) or (residuals[i]) > (mean + 3 * std):
                count += 1
                inds.append(i)

        # break if no points fall outside 3-sigma
        if count == 0:
            break
        iters += 1

        # save a record of removed data
        for x in inds:
            epochs_removed.append(epochs[x])
            obs_removed.append(observations[x])
            errs_removed.append(errs[x])

        # remove flagged data
        epochs = np.delete(epochs, inds)
        observations = np.delete(observations, inds)
        errs = np.delete(errs, inds)
        print(count, "removed")

        # repeat model fitting
        data = (epochs, observations, errs)
        data_fits = fitAll(data, niter, burn_in, planetinfo, save_directory)
        epochs, observations, errs = data_fits["DATA"]
        linear_model, decay_model, precession_model = data_fits["MODELS"]

    # save all removed data and number of iterations needed
    print("Epochs removed: ", epochs_removed)
    data_fits["REMOVED"] = [epochs_removed, obs_removed, errs_removed]
    data_fits["ITERS"] = iters

    return data_fits
