import csv
import json
import numpy as np
import astropy.time as time
import astropy.coordinates as coord
from astropy import units as u
from numpy import random

def calculateEpoch(t0, P, time, primary=True):
    """
    For calculating orbit number(s) if not given explicitly
        Primary=True if a transit, Primary=False if an occultation
    """
    if primary == True:
        E = int(round((time - t0)/P, 0))
    elif primary == False:
        E = int(round((time - t0 - P/2)/P, 0))
    else:
        print("error: invalid observation type")
    return E

def helio_to_bary(hjd, RA, DEC):
    """
    Utilizes astropy to convert HJD to BJD
    """
    helio = time.Time(hjd, scale='utc', format='jd')
    earthcentre = coord.EarthLocation(0., 0., 0.)
    coordinates = coord.SkyCoord(RA, DEC, frame='icrs')

    ltt = helio.light_travel_time(coordinates, 'heliocentric', location=earthcentre)
    guess = helio - ltt
    delta = (guess + guess.light_travel_time(coordinates, 'heliocentric', earthcentre)).jd - helio.jd
    guess -= delta * u.d

    ltt = guess.light_travel_time(coordinates, 'barycentric', earthcentre)
    BJD_TBD = guess.tdb + ltt
    BJD_TBD = np.array([float(x.to_value("jd")) for x in BJD_TBD])

    return BJD_TBD

def bary_to_helio(bjd, RA, DEC):
    """
    Utilizes astropy to convert BJD to HJD
    """
    bary = time.Time(bjd, scale='tdb', format='jd')
    earthcentre = coord.EarthLocation(0., 0., 0.)
    coordinates = coord.SkyCoord(RA, DEC, frame='icrs')

    ltt = bary.light_travel_time(coordinates, 'barycentric', location=earthcentre)
    guess = bary - ltt
    delta = (guess + guess.light_travel_time(coordinates, 'barycentric', earthcentre)).jd - bary.jd
    guess -= delta * u.d

    ltt = guess.light_travel_time(coordinates, 'heliocentric', earthcentre)
    HJD_UTC = guess.utc + ltt
    HJD_UTC = np.array([float(x.to_value("jd")) for x in HJD_UTC])

    return HJD_UTC

def readData(datafile):
    epochs = []
    observations = []
    errs = []
    types = []
    observers = []
    reader = csv.reader(open(datafile), delimiter="\t")
    for row in reader:
        epochs.append(int(row[1]))
        observations.append(float(row[2]))
        errs.append(float(row[3]))
        types.append("tra")     # Todo: change for inclusion of occultation data
        observers.append(row[5])
    epochs = np.array(epochs)
    observations = np.array(observations)
    errs = np.array(errs)
    return epochs, observations, errs, types

class NpEncoder(json.JSONEncoder):
    """
    Needed for saving results in correct format for json.dump
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def random_coin(a):
    """
    Used for MCMC scripts
    """
    uran = random.uniform(0,1)
    if uran < a:
        return True
    else:
        return False

def wrap(angle):
    """
    Simple angle wrapping
    """
    if angle < 0:
        angle += 2 * np.pi
    elif angle > 2 * np.pi:
        angle -= 2 * np.pi
    return angle

def modelCompare(model, observations, errors, variables):
    """
    Calculates chi square and Bayesian Information Criterion (BIC) for model comparison
    """
    chi2 = np.sum([(M - D) ** 2 / (err) ** 2 for M, D, err in zip(model, observations, errors)])
    bic = round(chi2 + len(variables) * np.log(len(observations)),1)
    return round(chi2,1), bic

def printFit(directory, results_linear, results_decay, results_precession=[0,0,0], fit_precess=False):
    """
    Prints best-fit parameters from model fitting
    """
    results_L, upper_L, lower_L  = results_linear
    results_D, upper_D, lower_D = results_decay
    results_P, upper_P, lower_P = results_precession

    # converts period derivative from per epoch to ms/yr
    conv = (365.25 * 24. * 3600. * 1e3)/results_D[1]

    with open(directory+"_fit_results.txt", "w") as f:
        f.write("LINEAR MODEL:"+"\n")
        f.write("t0 (transit): "+str(results_L[0])+" + "+str(upper_L[0])+" - "+str(lower_L[0])+"\n")
        f.write("P0 (transit): "+str(results_L[1])+" + "+str(upper_L[1])+" - "+str(lower_L[1])+"\n")
        f.write("" + "\n")
        f.write("DECAY MODEL:"+"\n")
        f.write("t0 (transit): "+str(results_D[0])+" + "+str(upper_D[0])+" - "+str(lower_D[0])+"\n")
        f.write("P0 (transit): "+str(results_D[1])+" + "+str(upper_D[1])+" - "+str(lower_D[1])+"\n")
        f.write("PdE (transit): "+str(results_D[2])+" + "+str(upper_D[2])+" - "+str(lower_D[2])+"\n")
        f.write("PdT (transit): "+str(results_D[2]*conv)+" + "+str(upper_D[2]*conv)+" - "+str(lower_D[2]*conv)+"\n")
        f.write("" + "\n")
        if fit_precess == True:
            f.write("PRECESSION MODEL:"+"\n")
            f.write("t0 (transit): "+str(results_P[0])+" + "+str(upper_P[0])+" - "+str(lower_P[0])+"\n")
            f.write("Ps (transit): "+str(results_P[1])+" + "+str(upper_P[1])+" - "+str(lower_P[1])+"\n")
            f.write("e (transit): "+str(results_P[2])+" + "+str(upper_P[2])+" - "+str(lower_P[2])+"\n")
            f.write("w0 (transit): "+str(results_P[3])+" + "+str(upper_P[3])+" - "+str(lower_P[3])+"\n")
            f.write("wdE (transit): "+str(results_P[4])+" + "+str(upper_P[4])+" - "+str(lower_P[4])+"\n")
    return

def readChains(directory, fit_precess=False):
    """
    Reads in posterior chains, written here to save space in plots.py
    """
    chain_L = []
    chain_file_L = directory + "_linear_burnedchain.txt"
    with open(chain_file_L) as infile_L:
        for line in infile_L:
            chain_L.append([float(i) for i in line.split(" ")])

    chain_D = []
    chain_file_D = directory + "_decay_burnedchain.txt"
    with open(chain_file_D) as infile_D:
        for line in infile_D:
            chain_D.append([float(i) for i in line.split(" ")])

    chain_P = []
    if fit_precess == True:
        chain_file_P = directory + "_precession_burnedchain.txt"
        with open(chain_file_P) as infile_P:
            for line in infile_P:
                chain_P.append([float(i) for i in line.split(" ")])

    return np.array(chain_L), np.array(chain_D), np.array(chain_P)

def table(settings_file):
    """
    Generates a summary of the
    """
    with open(settings_file) as json_file:
        settings = json.load(json_file)

    targets = settings["targets"]
    outfile = settings["save_directory"] + "results_table.txt"

    # generate empty arrays
    Pdots = []
    Pdot_errs = []
    BICs_linear = []
    BICs_decay = []
    BICs_precession = []

    for target in targets:
        # load results file
        results_directory = settings["save_directory"] + target + "/"
        with open(results_directory + target + "_results.json") as json_file:
            data = json.load(json_file)
        json_acceptable_string = data.replace("'", "\"")
        data = json.loads(json_acceptable_string)

        # read in data
        BIC_linear, BIC_decay, BIC_precess = data["BIC"]
        results_decay, lower_decay, upper_decay = data["DECAY"]

        # calculate Pdot
        conv = (365.25 * 24. * 3600. * 1e3) / results_decay[1]
        pdot = round(results_decay[2] * conv, 2)
        pdot_err = (round(lower_decay[2] * conv, 2), round(upper_decay[2] * conv, 2))

        # save values for this target
        Pdots.append(pdot)
        Pdot_errs.append(pdot_err)
        BICs_linear.append(BIC_linear)
        BICs_decay.append(BIC_decay)
        BICs_precession.append(BIC_precess)

    # make arrays and calculate difference in BIC
    BICs_linear = np.array(BICs_linear)
    BICs_decay = np.array(BICs_decay)
    BICs_precession = np.array(BICs_precession)
    diff = BICs_decay - BICs_linear

    # sort based on delta BIC
    # TODO: sorting by diff could cause issues if some targets have the same diff value (unlikely, but possible)
    targets = [x for _, x in sorted(zip(diff, targets))]
    Pdots = [x for _, x in sorted(zip(diff, Pdots))]
    Pdot_errs = [x for _, x in sorted(zip(diff, Pdot_errs))]
    BICs_linear = [x for _, x in sorted(zip(diff, BICs_linear))]
    BICs_decay = [x for _, x in sorted(zip(diff, BICs_decay))]
    BICs_precession = [x for _, x in sorted(zip(diff, BICs_precession))]
    diff = sorted(diff)

    Pdot_errs_T = np.array(Pdot_errs).T
    max_Pdot_errs = np.amax(Pdot_errs_T, axis=0).tolist()

    def round_sig(x, sig=2):
        return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

    for i in range(len(diff)):
        diff[i] = round(diff[i], 1)
        max_Pdot_errs[i] = str(round_sig(max_Pdot_errs[i], sig=2))
        Pdots[i] = str(Pdots[i])

    with open(outfile, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["Target","Decay Rate (ms/yr)", "1-sigma", "BIClinear", "BICdecay", "deltaBIC"])
        writer.writerows(zip(targets, Pdots, max_Pdot_errs, BICs_linear, BICs_decay, diff))