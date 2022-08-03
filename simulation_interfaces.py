import multiprocessing
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from json import loads


def plot_energy(sim_path, input_json_dict, p=None):
    df = pd.read_csv(f"{sim_path}/energy.dat", delimiter="\s+",names=['time', 'U','P','K'])
    STEPS = float(input_json_dict['steps'])
    # make sure our figure is bigger
    plt.figure(figsize=(15,3))
    # plot the energy
    plt.plot(df.time/0.002,df.U)
    # and the line indicating the complete run
    plt.ylim([-2,1])
    plt.plot([STEPS,STEPS],[df.U.max(),df.U.min()-2], color="r")
    plt.ylabel("Energy")
    plt.xlabel("Steps")
    which_sim = sim_path.split("/")[-1]
    plt.title(which_sim)
    if p is not None:
        print("Simulation is running:", p.is_alive())


def collect_freefiles(sim_dirs, n_bins):
    freefiles = {}
    sim_dirs_name = [sim_dir.split("/")[-2] for sim_dir in sim_dirs]
    for sim_dir, sim_dir_name, n_bin in zip(sim_dirs, sim_dirs_name, n_bins):
        freefile_dir = os.path.join(sim_dir, 'com_files', 'time_series', 'freefile')
        freefiles[sim_dir_name] = pd.read_csv(freefile_dir, sep='\t', nrows=int(n_bin))
    return freefiles


def plt_err(freefile, c, label, temp):
    temp = (temp + 273.15) / 3000
    plt.errorbar(0.8518 * freefile.loc[:, '#Coor'], (freefile.loc[:, 'Free'] / temp),
                 yerr=(freefile.loc[:, '+/-'] / temp), label=label, c=c, capsize=2, capthick=1.2, fmt='-',
                 linewidth=1.5, errorevery=5)


def plt_fig(title, xlabel, ylabel):
    plt.figure(dpi=150)
    plt.title(title, size=14)
    plt.xlabel(xlabel, size=10)
    plt.ylabel(ylabel, size=10)


def plot_free_energy(sim_dirs, n_bins, temps, seperate=False, c='tab:blue', label=None):
    freefiles = []
    sim_dirs_name = [sim_dir.split("/")[-2] for sim_dir in sim_dirs]
    for sim_dir, sim_dir_name, n_bin in zip(sim_dirs, sim_dirs_name, n_bins):
        freefile_dir = os.path.join(sim_dir, 'com_files', 'time_series', 'freefile')
        freefiles.append(pd.read_csv(freefile_dir, sep='\t', nrows=int(n_bin)))


    plt_fig('Free Energy', 'COM Distance (nm)', 'Free Energy (dG/kT)')
    label = sim_dirs_name
    for system, freefile in enumerate(freefiles):
        plt_err(freefile, c[system], label[system], temps[system])
        if seperate:
            plt_fig('Free Energy', 'COM Distance (nm)', 'Free Energy (dG/kT)')
    plt.legend()
    return None
