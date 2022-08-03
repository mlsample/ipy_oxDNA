import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import shutil
import fileinput
import sys
import subprocess


def copy_com_files(sim_dir, com_dir):
    # copy com files from each to window to separate directory
    if not os.path.exists(com_dir):
        os.makedirs(com_dir)
    windows = [w for w in os.listdir(sim_dir) if w.isdigit()]
    for window in windows:
        if os.path.isdir(os.path.join(sim_dir, window)):
            for file in os.listdir(os.path.join(sim_dir, window)):
                if 'com_distances' in file:
                    shutil.copyfile(os.path.join(sim_dir, window, file),
                                    os.path.join(com_dir, f'com_distances_{window}.txt'))
    return None


def collect_coms(com_dir):
    # create a list of dataframes for each window
    com_list = []
    com_files = [f for f in os.listdir(com_dir) if f.endswith('.txt')]
    for window, file in enumerate(com_files):
        com_list.append(pd.read_csv(os.path.join(com_dir, file), header=None, names=[window], usecols=[0]))
    return com_list


def autocorrelation(com_list):
    # create a list of autocorrelation values for each window
    autocorrelation_list = []
    for com in com_list:
        de = sm.tsa.acf(com, nlags=500)
        low = next(x[0] for x in enumerate(list(de)) if abs(x[1]) < (1 / np.e))
        autocorrelation_list.append(low)
    return autocorrelation_list


def number_com_lines(com_dir):
    # Create time_series directory and add com files with line number
    if os.path.exists(os.path.join(com_dir, 'time_series')):
        pass
    else:
        os.chdir(com_dir)
        if 'time_series' not in os.getcwd():
            shutil.copytree(os.getcwd(), os.path.join(com_dir, "time_series"))
        if 'time_series' in os.listdir(os.getcwd()):
            os.chdir(os.path.join(com_dir, "time_series"))
            files = os.listdir(os.getcwd())
            for file in files:
                for line in fileinput.input(file, inplace=True):
                    sys.stdout.write(f'{fileinput.filelineno()} {line}')
    time_dir = os.path.join(com_dir, 'time_series')
    return time_dir


def sort_coms(file):
    # Method to sort com files by window number
    var = int(file.split('_')[-1].split('.')[0])
    return var


def get_r0_list(xmin, xmax, sim_dir):
    # Method to get r0 list
    n_confs = len(os.listdir(sim_dir))
    r0_list = np.round(np.linspace(float(xmin), float(xmax), n_confs)[1:], 3)
    return r0_list


def create_metadata(time_dir, autocorrelation_list, r0_list, k):
    # Create metadata file to run WHAM analysis with
    os.chdir(time_dir)
    com_files = [file for file in os.listdir(os.getcwd()) if 'com_distances' in file]
    com_files.sort(key=sort_coms)
    with open(os.path.join(os.getcwd(), 'metadata'), 'w') as f:
        for file, r0, auto in zip(com_files, r0_list, autocorrelation_list):
            f.write(f'{file} {r0} {k} {auto}\n')
    return None


def run_wham(wham_dir, time_dir, xmin, xmax, n_bins, tol, n_boot, temp):
    # Run WHAM analysis on metadata file to create free energy file
    wham = os.path.join(wham_dir, 'wham')
    seed = str(np.random.randint(0, 1000000))
    os.chdir(time_dir)
    output = subprocess.run([wham, xmin, xmax, n_bins, tol, temp, '0', 'metadata', 'freefile', n_boot, seed], capture_output=True)
    return output

def format_freefile(time_dir):
    # Format free file to be readable by pandas
    os.chdir(time_dir)
    with open("freefile", "r") as f:
        lines = f.readlines()
    lines[0] = "#Coor\tFree\t+/-\tProb\t+/-\n"
    with open("freefile", "w") as f:
        for line in lines:
            f.write(line)
    return None

def wham_analysis(wham_dir, sim_dir, com_dir, xmin, xmax, k, n_bins, tol, n_boot, temp):
    print('Running WHAM analysis...')
    copy_com_files(sim_dir, com_dir)
    com_list = collect_coms(com_dir)
    autocorrelation_list = autocorrelation(com_list)
    time_dir = number_com_lines(com_dir)
    r0_list = get_r0_list(xmin, xmax, sim_dir)
    create_metadata(time_dir, autocorrelation_list, r0_list, k)
    output = run_wham(wham_dir, time_dir, xmin, xmax, n_bins, tol, n_boot, temp)
    print(output)
    format_freefile(time_dir)
    return output

