from umbrella_sampling import ComUmbrellaSampling, MeltingUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable
from wham_analysis import collect_coms
import os
import matplotlib.pyplot as plt
from vmmc import *
from collections import Counter
from parameter_search import * 
import numpy as np
from skopt import Optimizer
from skopt.space import Categorical, Space
from scipy.special import logsumexp
from copy import deepcopy

path = os.path.abspath('/scratch/matthew/ipy_oxDNA/ipy_oxdna_examples/ico_3p/')

system_name = 'inital_umbrella'

conditions = ['k05_xmax70_nwin100_45C']

systems = [f'{condition}' for condition in conditions]

file_dirs = [f'{path}/{system_name}' for _ in range(len(systems))]


monomer_1_patch_1 = '11674,11666,11667,11668,11669,11670,11671,11672,11673'
monomer_1_patch_2 = '12945,12953,12946,12947,12948,12949,12950,12951,12952'
monomer_1_patch_3 = '13320,13328,13321,13322,13323,13324,13325,13326,13327'

monomer_2_patch_1 = '6655,6663,6656,6657,6658,6659,6660,6661,6662'
monomer_2_patch_2 = '6358,6350,6351,6352,6353,6354,6355,6356,6357'
monomer_2_patch_3 = '5752,5744,5745,5746,5747,5748,5749,5750,5751'

monomer_1 = f'{monomer_1_patch_1},{monomer_1_patch_2},{monomer_1_patch_3}'
monomer_2 = f'{monomer_2_patch_1},{monomer_2_patch_2},{monomer_2_patch_3}'


xmin = 0
xmax = 70
n_windows = 100
starting_r0 = 1
stiff = 0.5
print_every = 1e4
temperature = '45C'
name = 'all_observables.txt'

pre_eq_steps = 1e7
eq_steps = 1e7
prod_steps = 2e8

verlet_skin = 0.5


pre_equlibration_parameters_list = [{'steps':f'{pre_eq_steps}','print_energy_every': f'{pre_eq_steps}',
                           'print_conf_interval':f'{pre_eq_steps}', 'refresh_vel': '1',
                           'fix_diffusion': '0', 'T':f'{temperature}', 'verlet_skin': f'{verlet_skin}' }
                            for _ in conditions]

equlibration_parameters_list = [{'steps':f'{eq_steps}','print_energy_every': f'{eq_steps}',
                           'print_conf_interval':f'{eq_steps}', 'refresh_vel': '1',
                           'fix_diffusion': '0', 'T':f'{temperature}', 'verlet_skin': f'{verlet_skin}'}
                            for _ in conditions]

production_parameters_list = [{'steps':f'{prod_steps}','print_energy_every': f'{prod_steps}',
                           'print_conf_interval':f'{prod_steps}', 'refresh_vel': '1',
                           'fix_diffusion': '0', 'T':f'{temperature}', 'verlet_skin': f'{verlet_skin}'}
                            for _ in conditions]

us_list = [MeltingUmbrellaSampling(file_dir, sys, clean_build='force') for file_dir, sys in zip(file_dirs,systems)]

simulation_manager = SimulationManager()

for us, pre_equlibration_parameters in zip(us_list, pre_equlibration_parameters_list):
    print(us.system)
    us.build_pre_equlibration_runs(simulation_manager, n_windows, monomer_1, monomer_2,
                               stiff, xmin, xmax, pre_equlibration_parameters, starting_r0, pre_eq_steps,
                               print_every=print_every, observable=True, protein=None,
                               force_file=None, continue_run=False, name=name)
    
simulation_manager.worker_manager(gpu_mem_block=False)

for us, equlibration_parameters in zip(us_list, equlibration_parameters_list):
    print(us.system)
    us.build_equlibration_runs(simulation_manager, n_windows, monomer_1, monomer_2,
                               stiff, xmin, xmax, equlibration_parameters, print_every=print_every,
                               observable=True, protein=None, force_file=None, continue_run=False, name=name)
    
simulation_manager.worker_manager(gpu_mem_block=False)

for us, production_parameters in zip(us_list, production_parameters_list):
    print(us.system)
    us.build_production_runs(simulation_manager, n_windows, monomer_1, monomer_2,
                             stiff, xmin, xmax, production_parameters,
                             observable=True, print_every=print_every ,protein=None,
                             force_file=None, continue_run=False, name=name)
    
simulation_manager.worker_manager(gpu_mem_block=False)

wham_dir = os.path.abspath('/scratch/matthew/ipy_oxDNA/src/wham/wham')
n_bins = '400'
tol = '1e-12'
n_boot = '0'


for us in us_list:
    us.wham_run(wham_dir, xmin, xmax, stiff, n_bins, tol, n_boot, all_observables=True)
    
n_chunks = 2
data_added_per_iteration = 3

for us in us_list:
    us.wham.get_n_data_per_com_file()
    us.convergence_analysis(n_chunks, data_added_per_iteration, wham_dir, xmin, xmax, stiff, n_bins, tol, n_boot)