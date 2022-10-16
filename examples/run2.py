from pathlib import Path

import numba as nb
import numpy as np
import scipy.stats as stats

from natsort import natsorted

import matplotlib.pyplot as plt


#plt.style.use("mike")
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')


import astropy.units as u

import cmasher as cmr

green = "#33FF86"
purple = "#CE33FF"


from cosmogrb.universe.survey import Survey

import os, sys
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from zusammen.stan_models.stan_model import get_model
from zusammen import AnalysisBuilder, DataSet
from zusammen.spectral_plot import display_posterior_model_counts

from threeML import update_logging_level

import arviz as av


update_logging_level("FATAL")


from astromodels import Band_Calderone, PointSource, Model


from threeML import JointLikelihood, DataList, display_spectrum_model_counts

from astromodels import Cutoff_powerlaw

import popsynth as ps

ds = DataSet.from_yaml("test_proc_sig_10_red.yml", mc_bound_limit=None)

import cmdstanpy
cmdstanpy.set_cmdstan_path("/home/bjorn/general_sw/cmdstan-2.29.0/")

m = get_model("cpl_simple_chunked_centered")

m.clean_model()

m.build_model()#opt_exp=True)

data = ds.to_stan_dict()
for i in range(data["N_intervals"]):
    stan_new = {}
    for key in data.keys():
        if key in ["N_intervals", "N_grbs"]:
            stan_new[key] = 1
        elif key in ["max_n_echan", "max_n_chan"]:
            stan_new[key] = data[key]
        else:
            stan_new[key] = data[key][i:i+1]
    n_threads = 4
    n_chains = 2

    fit = m.model.sample(
        data=stan_new,
        parallel_chains=n_chains,
        chains=n_chains,
        inits= {'alpha': -1.*np.ones(stan_new["N_intervals"]),
                "log_ec": 2*np.ones(stan_new["N_intervals"]),
                #"ec":100*np.ones(stan_new["N_intervals"]),
                "log_K": -1*np.ones(stan_new["N_intervals"]),
                #"log_energy_flux_mu_raw": 0*np.ones(data["N_intervals"]),
                #"log_energy_flux_sigma": 1,
                #"log_energy_flux_raw": 0,
                "log_energy_flux": -6*np.ones(stan_new["N_intervals"])},

        threads_per_chain=n_threads,
        seed=1234,
        iter_warmup=1000,
        iter_sampling=500,
        max_treedepth=15,
        adapt_delta=0.9,
        show_progress=True,
        #show_console=True
        output_dir=".stan/"
    )

    res = av.from_cmdstanpy(fit)

    res.to_netcdf(f"res/test_easy_{i}.nc")
