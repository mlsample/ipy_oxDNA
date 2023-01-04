import multiprocessing
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from json import loads
from scipy.stats import multivariate_normal


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


def plt_fig(title=None):
    from matplotlib.ticker import MultipleLocator
    plt.figure(dpi=200, figsize=(5.5, 4.5))
    plt.title(title)
    plt.xlabel('End-to-End Distance (nm)', size=12)
    plt.ylabel('Free Energy / k$_B$T', size=12)
    #plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['ytick.major.size'] = 6
    #plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['axes.linewidth'] = 1.25
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    ax = plt.gca()
    ax.set_aspect('auto')
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    #ax.yaxis.set_minor_locator(MultipleLocator(2.5))
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='minor', labelsize=9)
    ax.yaxis.set_ticks_position('both')
    return ax

def plt_err(system, ax, fmt='-', c=None, label=None):
    df = free_energy[system]
    ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Free'],
                 yerr=df.loc[:, '+/-'], label=label, c=c, capsize=2.5, capthick=1.2, fmt=fmt,
                 linewidth=1.5, errorevery=15)
    plot_indicator(system, w_means, ax, c)

def plot_indicator(system, indicator, ax, c=None):
    target = indicator[0]
    nearest = system.iloc[(system['#Coor'] -target).abs().argsort()[:1]]
    near_true = nearest
    x_val = near_true['#Coor']
    y_val = near_true['Free']
    ax.scatter(x_val, y_val, s=50, c=c, label=f'{target:.2f} nm \u00B1 {indicator[1]:.2f} nm')
    return None


def plot_free_energy(system, indicator=None, title='Free Energy Profile', c=None, label=None):
    ax = plt_fig(title=title)
    df = system
    ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Free'],
                 yerr=df.loc[:, '+/-'], label=label, c=c, capsize=2.5, capthick=1.2,
                 linewidth=1.5, errorevery=15)
    if indicator is not None:
        plot_indicator(system, indicator, ax, c)
    plt.legend()



def w_mean(df):
    free = df.loc[:, 'Free']
    coord = df.loc[:, '#Coor']
    prob = np.exp(-free) / sum(np.exp(-free))
    mean = sum(coord * prob)
    return mean


def bootstrap_w_mean_error(df):
    coord = df.loc[:, '#Coor']
    free = df.loc[:, 'Free'] 
    prob = np.exp(-free) / sum(np.exp(-free))

    err = df.loc[:, '+/-']
    mask = np.isnan(err)
    err[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), err[~mask])
    cov = np.diag(err**2)

    estimate = np.array(multivariate_normal.rvs(mean=free, cov=cov, size=100000, random_state=None))
    est_prob = [np.exp(-est) / sum(np.exp(-est)) for est in estimate]
    means = [sum(coord * e_prob) for e_prob in est_prob]
    standard_error = np.std(means)
    return standard_error


def to_si(path, system, temp, n_bins):
    free = pd.read_csv(f'{path}/{system}/umbrella_production/com_files/time_series/freefile', sep='\t', nrows=int(n_bins))
    free['Free'] = free['Free'].div(temp)
    free['+/-'] = free['+/-'].div(temp)
    free['#Coor'] *= 0.8518
    return free