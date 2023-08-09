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
    if os.path.exists(com_dir):
        shutil.rmtree(com_dir)
    if not os.path.exists(com_dir):
        os.mkdir(com_dir)
    windows = [w for w in os.listdir(sim_dir) if w.isdigit()]
    for window in windows:
        if os.path.isdir(os.path.join(sim_dir, window)):
            for file in os.listdir(os.path.join(sim_dir, window)):
                if 'com_distance' in file:
                    shutil.copyfile(os.path.join(sim_dir, window, file),
                                    os.path.join(com_dir, f'com_distance_{window}.txt'))
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
        de = sm.tsa.acf(com, nlags=50000)
        low = next(x[0] for x in enumerate(list(de)) if abs(x[1]) < (1 / np.e))
        if int(low) == 1:
            low = 2
        autocorrelation_list.append(low)
    return autocorrelation_list


def number_com_lines(com_dir):
    # Create time_series directory and add com files with line number
    if os.path.exists(com_dir):
        os.chdir(com_dir)
        files = os.listdir(os.getcwd())
        for file in files:
            for line in fileinput.input(file, inplace=True):
                sys.stdout.write(f'{fileinput.filelineno()} {line}')
    else:
        print('com_dir does not exist')
        return None
    return com_dir


# def number_com_lines(com_dir):
#     # Create time_series directory and add com files with line number
#     if os.path.exists(os.path.join(com_dir, 'time_series')):
#         pass
#     else:
#         os.chdir(com_dir)
#         if 'time_series' not in os.getcwd():
#             shutil.copytree(os.getcwd(), os.path.join(com_dir, "time_series"))
#         if 'time_series' in os.listdir(os.getcwd()):
#             os.chdir(os.path.join(com_dir, "time_series"))
#             files = os.listdir(os.getcwd())
#             for file in files:
#                 for line in fileinput.input(file, inplace=True):
#                     sys.stdout.write(f'{fileinput.filelineno()} {line}')
#     time_dir = os.path.join(com_dir, 'time_series')
#     return time_dir


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
    com_files = [file for file in os.listdir(os.getcwd()) if 'com_distance' in file]
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
    #print(output)
    format_freefile(time_dir)
    print('WHAM analysis completed')
    return output


def get_up_down(x_max:float, com_dist_file:str, pos_file:str):
    key_names = ['a', 'b', 'c', 'p', 'va', 'vb', 'vc', 'vp']
    def process_pos_file(pos_file:str , key_names:list) -> dict:
        cms_dict = {}
        with open(pos_file, 'r') as f:
            pos = f.readlines()
            pos = [line.strip().split(' ') for line in pos]
            for idx,string in enumerate(key_names):
                cms = np.transpose(pos)[idx]
                cms = [np.array(line.split(','), dtype=np.float64) for line in cms]
                cms_dict[string] = np.array(cms)
        return cms_dict
    
    def point_in_triangle(a, b, c, p):
        u = b - a
        v = c - a
        n = np.cross(u,v)
        w = p - a
        gamma = abs(np.dot(np.cross(u,w), n)) / np.dot(n,n)
        beta = abs(np.dot(np.cross(w,v), n)) / np.dot(n,n)
        alpha = 1 - gamma - beta
        return ((0 <= alpha) and (alpha <= 1) and (0 <= beta)  and (beta  <= 1) and (0 <= gamma) and (gamma <= 1))
    
    def point_over_plane(a, b, c, p):
        u = c - a
        v = b - a
        cp = np.cross(u,v)
        va, vb, vc = cp
        d = np.dot(cp, c)
        plane = np.array([va, vb, vc, d])
        point = np.array([p[0], p[1], p[2], 1])
        result = np.dot(plane, point)
        return 1 if result > 0 else 0
    
    def up_down(x_max:float, com_dist_file:str, pos_file:str) -> list:
        with open(com_dist_file, 'r') as f:
            com_dist = f.readlines()
        com_dist = [line.strip() for line in com_dist]
        com_dist = list(map(float, com_dist)) 
        cms_list = process_pos_file(pos_file, key_names)
        up_or_down = [point_in_triangle(a, b, c, p) for (a,b,c,p) in zip(cms_list['va'],cms_list['vb'],cms_list['vc'],cms_list['p'])]
        over_or_under = [point_over_plane(a, b, c, p) for (a,b,c,p) in zip(cms_list['va'],cms_list['vb'],cms_list['vc'],cms_list['vp'])]
        com_dist = [-state if direction == 0 else state for state, direction in zip(com_dist, over_or_under)]
        com_dist = [x_max - state if state > 0 else -x_max - state for state in com_dist]
        # if max(abs(max(com_dist)),  abs(min(com_dist))) > 18:
        #     com_dist = [dist if (np.sign(dist) == np.sign(np.mean(com_dist))) else -dist for dist in com_dist ]
        com_dist = [np.round(val, 4) for val in com_dist]
        return com_dist
    return(up_down(x_max, com_dist_file, pos_file))

def copy_pos_files(sim_dir, pos_dir):
    # copy com files from each to window to separate directory
    if os.path.exists(pos_dir):
        shutil.rmtree(pos_dir)
    if not os.path.exists(pos_dir):
        os.mkdir(pos_dir)
    windows = [w for w in os.listdir(sim_dir) if w.isdigit()]
    for window in windows:
        if os.path.isdir(os.path.join(sim_dir, window)):
            for file in os.listdir(os.path.join(sim_dir, window)):
                if 'cms_positions' in file:
                    shutil.copyfile(os.path.join(sim_dir, window, file),
                                    os.path.join(pos_dir, f'cms_positions_{window}.txt'))

def collect_pos(pos_dir):
    # create a list of dataframes for each window
    com_list = []
    com_files = [f for f in os.listdir(pos_dir) if f.endswith('.txt')]
    for window, file in enumerate(com_files):
        com_list.append(pd.read_csv(os.path.join(pos_dir, file), header=None, names=[window], usecols=[0]))
    return com_list

def mod_collect_coms(com_dir):
    # create a list of dataframes for each window
    com_list = []
    com_files = [f for f in os.listdir(com_dir) if f.endswith('.txt')]
    for window, file in enumerate(com_files):
        com_list.append(pd.read_csv(os.path.join(com_dir, file), header=None, names=None, usecols=[0]))
    return com_list


def get_r0_list_mod(xmin, xmax, sim_dir):
    # Method to get r0 list
    r0_list = np.round(np.linspace(float(xmin), float(xmax), n_confs)[1:], 3)
    return r0_list


def copy_com_pos(sim_dir, com_dir, pos_dir):
    copy_com_files(sim_dir, com_dir)
    copy_pos_files(sim_dir, pos_dir)
    return None

def get_xmax(com_dir_1, com_dir_2):
    com_list_1 = collect_coms(com_dir_1)
    com_list_2 = collect_coms(com_dir_2)
    xmax_1 = max([com.max().iloc[0] for com in com_list_1])
    xmax_2 = max([com.max().iloc[0] for com in com_list_2])
    xmax = max(xmax_1, xmax_2)
    return xmax

def modifed_coms(xmax, com_dir, pos_dir, mod_com_dir):
    if os.path.exists(mod_com_dir):
        shutil.rmtree(mod_com_dir)
    if not os.path.exists(mod_com_dir):
        os.mkdir(mod_com_dir)
    
    pos_files = [os.path.join(pos_dir,f) for f in os.listdir(pos_dir) if f.endswith('.txt')]
    com_files = [os.path.join(com_dir,f) for f in os.listdir(com_dir) if f.endswith('.txt')]
    pos_files.sort(key=sort_coms)
    com_files.sort(key=sort_coms)
    new_coms = []
    for pos_file, com_file in zip(pos_files, com_files):
        new_coms.append(get_up_down(xmax, com_file, pos_file))
    sign = np.sign(np.mean(new_coms))
    for idx in range(len(new_coms)):
        for val_id in range(len(new_coms[idx])):
            if np.sign(new_coms[idx][val_id]) != sign:
                if new_coms[idx][val_id] < 15:
                    new_coms[idx][val_id] = -new_coms[idx][val_id]
    
    # for idx in range(len(new_coms)):
    #     for val_id in range(len(new_coms[idx])):
    #         if sign == 1:
    #             if new_coms[idx][val_id] > 15:
    #                 if np.sign(new_coms[idx][val_id]) != sign:
    #                     new_coms[idx][val_id] = -new_coms[idx][val_id]
    for window, com in enumerate(new_coms):
        with open(os.path.join(mod_com_dir,'com_distance_'+str(window)+'.txt'), 'w') as f:
            for com_value in com:
                f.write(str(com_value)+'\n')
    return None

def mod_com_info(sim_dir, com_dir, pos_dir, mod_com_dir, xmax):
    modifed_coms(xmax, com_dir, pos_dir, mod_com_dir)
    mod_com_list = collect_coms(mod_com_dir)
    autocorrelation_list = autocorrelation(mod_com_list)
    time_dir = number_com_lines(mod_com_dir)
    return autocorrelation_list

def mod_create_metadata(mod_com_dir_1, mod_com_dir_2, auto_1, auto_2, r0_list_1, r0_list_2, k):
    # Create metadata file to run WHAM analysis with
    com_files_1 = [os.path.join(mod_com_dir_1,file) for file in os.listdir(mod_com_dir_1) if 'com_distance' in file]
    com_files_2 = [os.path.join(mod_com_dir_2,file) for file in os.listdir(mod_com_dir_2) if 'com_distance' in file]
    com_files_1.sort(key=sort_coms)
    com_files_2.sort(key=sort_coms)
    with open(os.path.join(mod_com_dir_1, 'metadata'), 'w') as f:
        for file, r0, auto in zip(com_files_1, r0_list_1, auto_1):
            f.write(f'{file} {r0} {k} {auto}\n')
    with open(os.path.join(mod_com_dir_1, 'metadata'), 'a') as f:
        for file, r0, auto in zip(com_files_2, r0_list_2, auto_2):
            f.write(f'{file} {r0} {k} {auto}\n')
    return None

def run_wham(wham_dir, time_dir, xmin, xmax, n_bins, tol, n_boot, temp):
    # Run WHAM analysis on metadata file to create free energy file
    wham = os.path.join(wham_dir, 'wham')
    seed = str(np.random.randint(0, 1000000))
    os.chdir(time_dir)
    output = subprocess.run([wham, xmin, xmax, n_bins, tol, temp, '0', 'metadata', 'freefile', n_boot, seed], capture_output=True)
    return output

#create function that takes in two autocorrelation_lists, two mod_com_dirs
def two_sided_wham(wham_dir, auto_1, auto_2, mod_com_dir_1, mod_com_dir_2, xmin, xmax, k, n_bins, tol, n_boot, temp):
    r0_list_1 = np.round(np.linspace(-float(xmax), 0, len(auto_1) +1), 3)[1:]
    r0_list_2 = np.round(np.linspace(float(0), float(xmax), len(auto_2)+1), 3)[:-1][::-1]
    mod_create_metadata(mod_com_dir_1, mod_com_dir_2, auto_1, auto_2, r0_list_2, r0_list_1, k)
    output = run_wham(wham_dir, mod_com_dir_1, str(-float(xmax)), xmax, n_bins, tol, n_boot, temp)
    format_freefile(mod_com_dir_1)
    print('WHAM analysis completed')


def cms_wham_analysis(wham_dir, mod_com_dir, xmin, xmax, k, n_bins, tol, n_boot, temp):
    r0_list = get_r0_list_mod(xmin, xmax, sim_dir)
    create_metadata(mod_com_dir, autocorrelation_list, r0_list, k)
    output = run_wham(wham_dir, mod_com_dir, xmin, xmax, n_bins, tol, n_boot, temp)
    format_freefile(mod_com_dir)
    print('WHAM analysis completed')
    return None




#I have two production folders
#I can create mod_pos files seperatly
#After I create mod_pos seperately
#I get the autocorrelation seperetly
# I can number the com lines seperately

# I need to input both umbrella simulations together.
# I can input both umbrella simulations in the umbrella interface?
#alternatively I would be doing that here
#If I did it here that would be one thing...
#either way I have to run the wham
#I think it would be intresting to do it here...
#but maybe not
#would would I return
# It would be better to do autocorrelation together
#if seperate I could just append and concatencate, not hard and would give me a return value
#maybe
#then I would have a wham analysis function that would take both autocorrelation,, but I need all the other things like the dir, but I would have that...
#So what Ill do is:
#Make a function that makes a mod_com folder and gets autocorrelation
#make a function that takes in 

#I have to create the r0 list together 