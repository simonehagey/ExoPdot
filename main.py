import utils
import os
import errno
import json
import plots
import fitModels

"""
This is the main script for running the analysis that compares the constant period, orbital decay, and
apsidal precession transit timing models. All relevant settings can be changed in the "settings-example.json" file.
"""

# open settings file
settings_file = "settings-example.json"
with open(settings_file) as json_file:
    settings = json.load(json_file)

# read in settings
targets = settings["targets"]   # do you want to run sigma-clipping algorithm (1-yes, 0-no)
fit_precess = settings["fit_precession"]    # do you want to fit the precession model (1-yes, 0-no)
all_plots = settings["diagnostic_plots"]
iterate = settings["sigma_clip"]   # list of exoplanet target names
niter = settings["niter"]   # number of MCMC iterations (linear, decay, precession)
burn_in = settings["burn_in"]   # number of MCMC iterations to discard (linear, decay, precession)

# iterate through all systems
for target in targets:

    datafile = settings["data_directory"] + target + settings["data_subscript"]
    planetfile = settings["planet_directory"] + target
    save_directory = settings["save_directory"] + target + "/"
    if not os.path.exists(save_directory):
        try:
            os.makedirs(save_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # load planetary system info
    with open(planetfile) as json_file:
        planetinfo = json.load(json_file)

    # load data
    epochs, observations, errs, types = utils.readData(datafile)
    data = (epochs, observations, errs)

    # do an initial fit of just constant period and orbital decay models
    initial_fit = fitModels.fitAll(data, [100000, 100000, 0], [10000, 10000, 0], planetinfo, save_directory, False, all_plots)
    epochs, observations, errs = initial_fit["DATA"]
    data = (epochs, observations, errs)

    # running the sigma clipping algorithm (if desired)
    max_iters = 20
    if iterate == True:
        iter_fit = fitModels.iterate(initial_fit, max_iters, [100000, 100000, 0], [10000, 10000, 0], planetinfo, save_directory)
        epochs, observations, errs = iter_fit["DATA"]
        data = (epochs, observations, errs)

    # do a final fit including apsidal precession model (if desired)
    fit_results_final = fitModels.fitAll(data, niter, burn_in, planetinfo, save_directory, fit_precess, all_plots)

    # add MCMC info to results file
    fit_results_final["BURN"] = burn_in
    fit_results_final["NITER"] = niter
    if iterate == True:
        fit_results_final["ITERS"] = iter_fit["ITERS"]
        fit_results_final["REMOVED"] = iter_fit["REMOVED"]

    # save results to .json file
    fits = json.dumps(fit_results_final, cls=utils.NpEncoder)
    with open(save_directory + target + "_results.json", 'w') as outfile:
        json.dump(fits, outfile)

# generate O-C (timing residuals) plot
plots.OC(settings_file)
# generate table of results from all targets
utils.table(settings_file)