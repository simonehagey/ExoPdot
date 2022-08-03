import utils
import json
import numpy as np
import linearFit
import decayFit
import precessionFit
from matplotlib import pyplot as plt

def OC(settings_file):
    """
    Generates O-C plots seen in Hagey et al. 2022. Modifications can be made in the settings
    file for varied datasets
    """
    with open(settings_file) as json_file:
        settings = json.load(json_file)

    # read in settings
    targets = settings["targets"]
    fit_precess = settings["fit_precession"]
    iterate = settings["sigma_clip"]
    minE = settings["pre_baseline_extension"]
    maxE = settings["post_baseline_extension"]
    ylim = settings["plot_ylimits"]

    for target in targets:
        save_directory = settings["save_directory"] + target + "/"

        # load results file
        with open(save_directory + target + "_results.json") as json_file:
            data = json.load(json_file)
        json_acceptable_string = data.replace("'", "\"")
        data = json.loads(json_acceptable_string)

        # read in data and model fitting results
        epochs, observed, errs = data["DATA"]
        linear, decay, precess = data["MODELS"]
        BIC_linear, BIC_decay, BIC_precess = data["BIC"]
        results_linear, lower_linear, upper_linear = data["LINEAR"]
        results_decay, lower_decay, upper_decay = data["DECAY"]
        if iterate == True:
            epochs_removed, observed_removed, errs_removed = data["REMOVED"]
        if fit_precess == True:
            results_precess, lower_precess, upper_precess = data["PRECESSION"]

        # generate models and calculate timing residuals
        epochs_full = np.arange(min(epochs) - minE, max(epochs) + maxE, 1)
        transit_OCs = np.array(observed) - np.array(linear)
        linear_model = linearFit.model(results_linear, epochs_full)
        decay_model = decayFit.model(results_decay, epochs_full)
        linear_OCs = (linear_model - linear_model)
        decay_OCs = (decay_model - linear_model)
        if fit_precess == True:
            precession_model = precessionFit.model(results_precess, epochs_full)
            precess_OCs = (precession_model - linear_model)
        if iterate == True:
            transit_OCs_removed = np.array(observed_removed) - linearFit.model(results_linear, epochs_removed)

        # read in chains
        chain_L, chain_D, chain_P = utils.readChains(save_directory+target, fit_precess)

        # generate indices for random samples of posterior chains
        inds_L = np.random.randint(len(chain_L), size=100)
        inds_D = np.random.randint(len(chain_D), size=100)
        if fit_precess == True:
            inds_P = np.random.randint(len(chain_P), size=150)


        """
        PLOT
        """
        plt.rcParams.update(settings["figure_params"])

        # plot samples from posterior chains
        if fit_precess == True:
            for ind in inds_P:
                sample = chain_P[ind]
                precession_model_sample = precessionFit.model(sample, epochs_full)
                thing = precession_model_sample - linear_model
                plt.plot(epochs_full, np.array(thing) * 1440, linewidth=1.5, color="cadetblue", alpha=0.2)

        for ind in inds_L:
            sample = chain_L[ind]
            linear_model_sample = linearFit.model(sample, epochs_full)
            thing = linear_model_sample - linear_model
            plt.plot(epochs_full, np.array(thing) * 1440, linewidth=1.5, color="darkgrey", alpha=0.05)

        for ind in inds_D:
            sample = chain_D[ind]
            decay_model_sample = decayFit.model(sample, epochs_full)
            thing = decay_model_sample - linear_model
            plt.plot(epochs_full, np.array(thing) * 1440, linewidth=1.5, color="#9c3535", alpha=0.05)

        # plot best-fit models
        plt.plot(epochs_full, np.array(linear_OCs) * 1440, label="Constant Period"+" (BIC="+str(BIC_linear)+")",
                 linewidth=.1, color="darkgrey", alpha=1)
        plt.plot(epochs_full, np.array(decay_OCs) * 1440, label="Orbital Decay"+" (BIC="+str(BIC_decay)+")",
                 linewidth=.1, color="#9c3535", alpha=1)
        if fit_precess == True:
            plt.plot(epochs_full, np.array(precess_OCs) * 1440, label="Apsidal Precession"+" (BIC="+str(BIC_precess)+")",
                     linewidth=.1, color="cadetblue", alpha=1)

        # plot data
        plt.errorbar(epochs, np.array(transit_OCs) * 1440, yerr=np.array(errs) * 1440,fmt='.',markersize='10',
                     ecolor='#595e66', elinewidth=1, color="k")
        # plot data removed in sigma clipping (if performed)
        if iterate == True:
            plt.errorbar(epochs_removed, np.array(transit_OCs_removed)*1440, yerr=np.array(errs_removed)*1440,
                         label="Excluded",fmt='x',markersize='5', ecolor='r', elinewidth=.8, color="r")

        # plot reference lines for 2023, 2025, and 2030
        text_x_pos = -70
        text_y_pos = ylim[0]+5

        t2023 = 2459945         # Calculate closest epoch to January 1st of each year
        e2023 = int((t2023 - results_linear[0])/results_linear[1])
        t2025 = 2460676
        e2025 = int((t2025 - results_linear[0])/results_linear[1])
        t2030 = 2462502
        e2030 = int((t2030 - results_linear[0])/results_linear[1])

        plt.axvline(x=e2023, linewidth=0.5, linestyle="-", color="grey")
        plt.axvline(x=e2025, linewidth=0.5, linestyle="-", color="grey")
        plt.axvline(x=e2030, linewidth=0.5, linestyle="-", color="grey")
        plt.text(e2023+text_x_pos,text_y_pos,"2023",rotation=90,fontsize=14)
        plt.text(e2025+text_x_pos,text_y_pos,"2025",rotation=90,fontsize=14)
        plt.text(e2030+text_x_pos,text_y_pos,"2030",rotation=90,fontsize=14)

        # finish plot
        conv = (365.25 * 24. * 3600. * 1e3) / results_decay[1]
        plt.title(target + " b | " + str(round(results_decay[2]*conv,2))
                  +"$^{+"+str(round(upper_decay[2]*conv,2))+"}_{-"+str(round(lower_decay[2]*conv,2))+"}$ ms/yr", fontsize=18)

        plt.xlabel("Epoch",labelpad=10)
        plt.ylabel("Timing Deviation (minutes)")
        plt.xlim(epochs_full[0],epochs_full[-1])
        plt.ylim(ylim[0], ylim[1])

        legend = plt.legend(loc="upper right",borderpad=1.03)
        for line in legend.get_lines():
            line.set_linewidth(6)

        plt.savefig(save_directory+target+"_O-C")
        plt.close()
