
#utilis
from oxdna_json_utils import input_json
from oxdna_writing_utils import *

#Simulation parameters
from forces import Force

#Write simulation function
from simulation_types import Simulation
from production_files import *
from analysis import wham_analysis

#other imports
import os
import json as js
import sys
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy


def multi_walker_equilibration(path, system, input_json_dict, n_walkers=1):
    processes = {}
    for walker in range(n_walkers):
        input_json_dict['CUDA_device'] = str(walker)
        input_eq = deepcopy(input_json_dict)
        processes[str(walker)] = Simulation.equilibration(
            file_dir=f"{path}/{system}",
            sim_dir=f"{path}/{system}/eq_{walker}",
            input_json_dict=input_eq,
            auto_run=True
        )
    return processes


def multi_walker_force_equilibration(path, system, input_json_dict, walker_forces, n_walkers=1):
    processes = {}

    input_json_dict["external_forces"] = "1"
    input_json_dict["external_forces_file"] = f"forces.json"

    for walker in range(n_walkers):
        input_json_dict['CUDA_device'] = str(walker)
        input_force_eq = deepcopy(input_json_dict)
        processes[str(walker)] = Simulation.force_equilibration(
            file_dir=f"{path}/{system}/eq_{walker}",
            sim_dir=f"{path}/{system}/r0_eq_{walker}",
            all_forces=walker_forces[str(walker)],
            input_json_dict=input_force_eq,
            auto_run=True
        )
    return processes


def multi_walker_pulling(path, system, input_json_dict, walker_forces, walker_steps_per_conf, n_walkers=1):
    processes = {}

    input_json_dict["external_forces"] = "1"
    input_json_dict["external_forces_file"] = f"forces.json"

    for walker in range(n_walkers):
        input_json_dict['CUDA_device'] = str(walker)
        input_pull = deepcopy(input_json_dict)
        processes[str(walker)] = Simulation.com_pulling(
            file_dir=f"{path}/{system}/r0_eq_{walker}",
            sim_dir=f"{path}/{system}/pull_{walker}",
            all_forces=walker_forces[str(walker)],
            n_windows=100,
            xmin=0,
            xmax=8,
            input_json_dict=input_pull,
            steps_per_conf=walker_steps_per_conf[walker],
            run_file=None,
            auto_run=True
        )
    return processes


def prepare_melting_umbrella(path, system, input_json_dict, all_forces, steps_per_conf, xmin, xmax, n_windows):

    input_eq = deepcopy(input_json_dict)

    eq_process = Simulation.equilibration(
        file_dir=f"{path}/{system}",
        sim_dir=f"{path}/{system}/{system}_eq",
        input_json_dict=input_eq,
        auto_run=True
    )
    eq_process.join()

    input_json_dict["external_forces"] = "1"
    input_json_dict["external_forces_file"] = f"forces.json"
    input_force_eq = deepcopy(input_json_dict)

    f_eq_process = Simulation.force_equilibration(
        file_dir=f"{path}/{system}/{system}_eq",
        sim_dir=f"{path}/{system}/{system}_r0_eq",
        all_forces=all_forces,
        input_json_dict=input_force_eq,
        auto_run=True
    )
    f_eq_process.join()

    input_pull = deepcopy(input_json_dict)
    pull_process = Simulation.com_pulling(
        file_dir=f"{path}/{system}/{system}_r0_eq",
        sim_dir=f"{path}/{system}/{system}_pull",
        all_forces=all_forces,
        n_windows=n_windows,
        xmin=xmin,
        xmax=xmax,
        input_json_dict=input_pull,
        steps_per_conf=steps_per_conf,
        run_file=None,
        auto_run=True
    )
    pull_process.join()
    return None


def multi_prepare_melting_umbrella(path, systems, input_json_dict, walker_forces, steps_per_conf, xmin, xmax, n_windows, n_walkers=1):
    processes = {}
    for walker in range(n_walkers):
        input_json_dict[walker]['CUDA_device'] = str(walker)
        walker_input_json_dict = deepcopy(input_json_dict)
        processes[f'{walker}'] = spawn(prepare_melting_umbrella, (path, systems[walker], walker_input_json_dict[walker], walker_forces[str(walker)], steps_per_conf[walker], xmin[walker], xmax[walker], n_windows[walker]))
    return processes


def equlibration_umbrella_setup(path, system, p1, p2, input_json_dict, all_forces, xmin, xmax, n_gpus, run_file):
    input_json_dict["external_forces"] = "1"
    input_json_dict["external_forces_file"] = f"forces.json"
    input_json_dict["observables_file"] = "observables.json"
    input_production = deepcopy(input_json_dict)
    write_production(
        file_dir=f"{path}/{system}/{system}_pull",
        sim_dir=f"{path}/{system}/umbrella_equlibration",
        p1=p1,
        p2=p2,
        xmin=xmin,
        xmax=xmax,
        all_forces=all_forces,
        n_gpus=n_gpus,
        input_json_dict=input_production,
        run_file=run_file
    )
    return None


def multi_umbrella_equlibration_setup(path, systems, p1, p2, walker_input_json_dict, walker_forces, xmin, xmax, n_gpus, run_file, n_walkers=1):
    processes = {}
    for walker in range(n_walkers):
        processes[f'{walker}'] = spawn(equlibration_umbrella_setup, (path, systems[walker], p1[walker], p2[walker], walker_input_json_dict[walker], walker_forces[str(walker)], xmin[walker], xmax[walker], n_gpus[walker], run_file))
        processes[f'{walker}'].join()
    return processes


def umbrella_production_setup(path, system, input_json_dict, n_gpus, n_windows, run_file):
    #copy over directories from umbrella equlibration and save them to umbrella production
    #the difference between the two should be nearly none
    #the dat files should be last conf files
    #the number of steps in the input should change
    #the equlibration does not need an observable

    input_json_dict["external_forces"] = "1"
    input_json_dict["external_forces_file"] = f"forces.json"
    input_json_dict["observables_file"] = "observables.json"
    input_production = deepcopy(input_json_dict)
    file_dir=f"{path}/{system}/umbrella_equlibration"
    sim_dir=f"{path}/{system}/umbrella_production"
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)    
    
    w_p_gpu = get_windows_per_gpu(n_gpus, n_windows)
    
    for window in range(n_windows):        
        prod_win = os.path.join(sim_dir, str(window))
        if not os.path.exists(prod_win):
            os.mkdir(prod_win)
    for window in range(n_windows):        
        eq_win = os.path.join(file_dir, str(window))
        prod_win = os.path.join(sim_dir, str(window))

        
        dat, top = get_last_conf_top(eq_win)
        
        shutil.copy(os.path.join(eq_win, 'last_conf.dat'), prod_win)
        
        shutil.copy(os.path.join(eq_win, top), prod_win)
        
        try:
            shutil.copy(os.path.join(eq_win, 'oxDNA2_sequence_dependent_parameters.txt'), prod_win)
        except:
            pass
        
        shutil.copy(os.path.join(eq_win, 'observables.json'), prod_win)
        
        shutil.copy(os.path.join(eq_win, 'forces.json'), prod_win)
        
        shutil.copy(os.path.join(eq_win, 'run.sh'), prod_win)
        
        input_json_dict["topology"] = top
        write_input(prod_win, input_production)

        write_production_run_file(run_file, w_p_gpu, sim_dir)
        
            
    print(f'{n_windows} umbrella windows created')
    return None


def multi_umbrella_production_setup(path, systems, walker_input_json_dict, n_gpus, n_windows, run_file, n_walkers=1):
    processes = {}
    for walker in range(n_walkers):
        processes[f'{walker}'] = spawn(umbrella_production_setup, (path, systems[walker], walker_input_json_dict, n_gpus[walker], n_windows[walker], run_file))
        processes[f'{walker}'].join()
    return processes


def production_setup(path, system, p1, p2, input_json_dict, all_forces, xmin, xmax, n_gpus, run_file):
    input_json_dict["external_forces"] = "1"
    input_json_dict["external_forces_file"] = f"forces.json"
    input_json_dict["observables_file"] = "observables.json"
    input_production = deepcopy(input_json_dict)
    write_production(
        file_dir=f"{path}/{system}/{system}_pull",
        sim_dir=f"{path}/{system}/umbrella_production_{input_production['steps']}",
        p1=p1,
        p2=p2,
        xmin=xmin,
        xmax=xmax,
        all_forces=all_forces,
        n_gpus=n_gpus,
        input_json_dict=input_production,
        run_file=run_file
    )
    return None


def multi_production_setup(path, systems, p1, p2, walker_input_json_dict, walker_forces, xmin, xmax, n_gpus, run_file, n_walkers=1):
    processes = {}
    for walker in range(n_walkers):
        processes[f'{walker}'] = spawn(production_setup, (path, systems[walker], p1, p2, walker_input_json_dict, walker_forces[str(walker)], xmin[walker], xmax[walker], n_gpus[walker], run_file))
        processes[f'{walker}'].join()
    return processes


def prepare_umbrella_sampling(path, system, p1, p2, input_json_dict, all_forces, steps_per_conf, xmin, xmax, n_windows, n_gpus, run_file):
    input_eq = deepcopy(input_json_dict)

    eq_process = Simulation.equilibration(
        file_dir=f"{path}/{system}",
        sim_dir=f"{path}/{system}/{system}_eq",
        input_json_dict=input_eq,
        auto_run=True
    )
    eq_process.join()

    input_json_dict["external_forces"] = "1"
    input_json_dict["external_forces_file"] = f"forces.json"
    input_force_eq = deepcopy(input_json_dict)

    f_eq_process = Simulation.force_equilibration(
        file_dir=f"{path}/{system}/{system}_eq",
        sim_dir=f"{path}/{system}/{system}_r0_eq",
        all_forces=all_forces,
        input_json_dict=input_force_eq,
        auto_run=True
    )
    f_eq_process.join()

    input_pull = deepcopy(input_json_dict)
    pull_process = Simulation.com_pulling(
        file_dir=f"{path}/{system}/{system}_r0_eq",
        sim_dir=f"{path}/{system}/{system}_pull",
        all_forces=all_forces,
        n_windows=n_windows,
        xmin=xmin,
        xmax=xmax,
        input_json_dict=input_pull,
        steps_per_conf=steps_per_conf,
        run_file=None,
        auto_run=True
    )
    pull_process.join()

    input_json_dict["external_forces"] = "1"
    input_json_dict["external_forces_file"] = f"forces.json"
    input_json_dict["observables_file"] = "observables.json"
    input_production = deepcopy(input_json_dict)
    write_production(
        file_dir=f"{path}/{system}/{system}_pull",
        sim_dir=f"{path}/{system}/umbrella_production_{input_production['steps']}",
        p1=p1,
        p2=p2,
        xmin=xmin,
        xmax=xmax,
        all_forces=all_forces,
        n_gpus=n_gpus,
        input_json_dict=input_production,
        run_file=run_file
    )
    return None


def multi_prepare_umbrella_sampling(path, systems, p1, p2, input_json_dict, walker_forces, steps_per_conf, xmin, xmax, n_windows, n_gpus, run_file, n_walkers=1):
    processes = {}
    for walker in range(n_walkers):
        input_json_dict['CUDA_device'] = str(walker)
        walker_input_json_dict = deepcopy(input_json_dict)
        processes[f'{walker}'] = spawn(prepare_umbrella_sampling(path, systems[walker], p1, p2, input_json_dict, walker_forces[str(walker)], steps_per_conf[walker], xmin[walker], xmax[walker], n_windows[walker], n_gpus[walker], run_file))
    return processes


def preform_umbrella_sampling(path, system, p1, p2, input_json_dict, all_forces, steps_per_conf, xmin, xmax, n_windows, n_gpus, run_file, sim_dir):
    prepare_umbrella_sampling(path, system, p1, p2, input_json_dict, all_forces, steps_per_conf, xmin, xmax, n_windows, n_gpus, run_file)
    run_production_slurm_files(n_gpus, n_windows, sim_dir)


def multi_analysis(wham_dir, sim_dir, com_dir, xmin, xmax, k, n_bins, tol, n_boot, temp, n_walkers=1):
    processes = {}
    for walker in range(n_walkers):
        processes[f'{walker}'] = spawn(wham_analysis, (wham_dir, sim_dir[walker], com_dir[walker], xmin[walker], xmax[walker], k[walker], n_bins[walker], tol[walker], n_boot[walker], temp[walker]))
        processes[f'{walker}'].join()
    return processes


