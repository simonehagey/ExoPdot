# ExoPdot

This repository contains source code, data, and a summary of results from *"Evidence of Long-Term Period Variations in the 
Exoplanet Transit Database (ETD)"* paper by Hagey, Edwards, and Boley (2022).

The main function of this code is to process the transit center 
times provided in the ~/DATA/ folder and compare three different transit timing models: 
(1) a planet on a constant orbital period, (2) a planet on a decaying, circular orbit, and (3) a planet that has a constant 
orbital period, but the orbit is precessing. The best-fit models are found through custom Metropolis-Hastings MCMC routines 
(with a Gibbs sampler) and compared via the Bayesian Information Criterion (BIC). As described in the paper, 
an (optional) iterative sigma-clipping algorithm is incorporated into the pipeline to handle the exclusion of spurious data points. 

### Data

All data used for this paper is from the [Exoplanet Transit Database (ETD)](http://var2.astro.cz/ETD/index.php).
If you are using it for your research please be sure to cite the database appropriately.
 
Data of all 30 star-planet systems investigated in the study are available in tab-separated .txt files in the ~/DATA/ folder. 
For the top 10 targets of interest (see paper) we provide datasets that have been manually cleaned of partial 
transits and duplicate submissions. The columns contain the ETD submission number, epoch, transit center (HJD), 
uncertainty (days), data quality (DQ) factor, and source/observer.

```
#   Epoch   T_mid (HJD)    Unc. (days)  DQ  Source
1   0       2455528.86774   0.00014	1   Hellier et al. 2011
4   497     2455933.16395   0.00025	1   Starr P.
5   505	    2455939.67397   0.00052	2   Naves R.
...
```

## Getting started

The code relies on the following packages that can be installed via pip: [astropy](https://github.com/astropy/astropy),
[corner](https://github.com/dfm/corner.py),
[matplotlib](https://github.com/matplotlib/matplotlib),
[numpy](https://github.com/numpy/numpy),
[scipy](https://github.com/scipy/scipy),
 and [tqdm](https://github.com/tqdm/tqdm).
 
Then clone this repository to your local machine:
```
git clone https://github.com/simonehagey/ExoPdot.git
```
Immediately upon cloning the repository and installing the required packages a simple analysis of the WASP-12 b data 
can performed by running the *main.py* script.
```
python3 main.py
```

The repository contains 7 Python scripts in addition to the transit timing data and analysis results. The functionality 
is designed to be fully encapsulated in the *main.py* script, so changes do not need to be made to other files unless 
you want to experiment with adjusting the bounds on parameters in the MCMC routines in the *fitModels.py* script.

### Settings file

Inside *main.py* you may specify the name of the "settings" .json file that define the planets, directories, and model 
fitting and plotting parameters. The default is *settings-example.json* which runs a simple WASP-12 b example. 

The first section of the file defines a list of planets to analyse (can be more than one), the directories containing the 
transit timing data and planetary system info files (see below), the data subscript (eg. change to "_ETD_og.txt" to use data pre-cleaning),
and directory where the results are to be saved. 
```
{"targets":["WASP-12"],
  "data_directory":"DATA/ETD/cleaned/",
  "data_subscript":"_ETD_clean.txt",
  "save_directory":"RESULTS/",
  "planet_directory":"SYSTEM_FILES/",
...
```
The second section of the file allows one to choose (1-yes, 0-no) whether to perform the sigma-clipping routine and/or fit the apsidal
precession model. The precession model is not chosen by default for time considerations. Additionally, the desired number of 
MCMC iterations (total) and number of burn-in iterations can be defined for each model (in the order: linear, decay, precession). 
Note that for all iterations of the (optional) sigma-clipping routine, the number of MCMC iterations is hardcoded to 100,000
for efficiency, so in the settings file the user is selecting the number of iterations for the final model fit at the 
end of the routine.

```
...
  "_comment1":"settings for model fitting",
  "sigma_clip":1,
  "fit_precession":0,
  "niter":[500000,500000,500000],
  "burn_in":[50000,50000,50000],
...
```

### System Info Files
All of the necessary information (and more) on the 30 star-planet systems investigated in this study are contained in separate
files in the ~/SYSTEM_FILES/ directory. For this project, only the reference transit time "tc", orbital period "p", and coordinates "RA"
and "DEC" were used. This system will be modified in future iterations to be a single file with updated orbital elements.

## Outputs

Running *main.py* automatically generates - for every planet - an O-C plot, text file displaying the printed 
model fitting and comparison results, saved copies of the MCMC posterior chains, and a "results" .json file. This
file contains the necessary outputs for interpreting the results such as the data, best-fit models, and BIC values, as
well as records from the MCMC routines such as the number of iterations, burn-in, allowed parameter ranges, and more. 
As a .json file, it is structured like a Python dictionary object, the most critical "keys" being:

* "DATA": an array (3,N) of the epochs, observations, and errors
* "MODELS": an array (3,N) of the predicted transit times from the linear, quadratic, and precession models
* "REMOVED": an array (3,N) of the epochs, observations, and errors removed during the sigma-clipping routine
* "BIC": a list of the BIC statistic for the best-fit linear, quadratic, and precession models
* "LINEAR": an array (3,2) of the best-fit linear model parameters (t0, P) with lower and upper bounds (1-sigma)
* "DECAY": an array (3,3) of the best-fit quadratic model parameters (t0, P, PdE) with lower and upper bounds (1-sigma)
* "PRECESSION": an array (3,5) of the best-fit precession model parameters (t0, P, e, w0, wdE) with lower and upper bounds (1-sigma)

See the *plots.py* script for an example of how to access and use this information if you would like to explore the results
further than the automatically generated files allow.

![Image](./RESULTS/ETD_paper/WASP-12/WASP-12_O-C.png)

```
LINEAR MODEL:
t0 (transit): 2454508.9780972297 + 4.525342956185341e-05 - 4.585832357406616e-05
P0 (transit): 1.0914195006314042 + 2.0790877908183347e-08 - 1.9656896332875817e-08

DECAY MODEL:
t0 (transit): 2454508.976206188 + 7.696636021137238e-05 - 7.622223347425461e-05
P0 (transit): 1.091422168511235 + 8.294883624415661e-08 - 8.65025684415599e-08
PdE (transit): -1.1861151993575246e-09 + 3.797866863275066e-11 - 3.570294672699221e-11
PdT (transit): -34.295573330989846 + 1.098122860084232 - 1.0323221793896804

PRECESSION MODEL:
t0 (transit): 2454508.969034184 + 0.0008613136596977711 - 0.0006090821698307991
Ps (transit): 1.0914194962752275 + 2.0686910184863905e-08 - 2.061203518977095e-08
e (transit): 0.029292425817275067 + 0.001761396102335274 - 0.002440446204276449
w0 (transit): 2.3606089473474796 + 0.026076921121455765 - 0.041118312338838336
wdE (transit): 0.00034656071608454137 + 1.7773255430349514e-05 - 1.0997649733655661e-05

MODEL COMPARISON:
Linear model: chi^2 = 2439.0  BIC = 2449.7
Decay model: chi^2 = 1911.2  BIC = 1927.3
Precession model: chi^2 = 1920.5  BIC = 1947.3
```




Developed by [Simone Hagey](mailto:shagey@student.ubc.ca)