from ipy_oxdna.umbrella.umbrella_sampling import MeltingUmbrellaSampling
from ipy_oxdna.oxdna_simulation import SimulationManager

import numpy as np
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex', 'bright'])


path = os.path.abspath('./')

file_dir_name = 'oxdna_files'  #'NAME_OF_DIR_CONTAINING_OXDNA_DAT_TOP'
file_dir = join(path, file_dir_name) #f'{path}/{file_dir_name}'

sim_dir_name = '8_nt_duplex_melting_tutorial' #'CHOSEN_NAME_WHERE_SIMULATION_WILL_BE_STORED'
sim_dir = join(file_dir, sim_dir_name)   #f'{path}/{sim_dir_name}'

# Initialize the umbrella sampling object, this will create the simulation directory

# Clean build has 3 options, False, True and 'force' False will not allow you to overwrite
# existing simulation directories, True will overwrite existing simulation directories after asking for confirmation
# 'force' will overwrite existing simulation directories without asking for confirmation
us = MeltingUmbrellaSampling(file_dir, sim_dir, clean_build=True)

# The simulation manager object is used to run the simulation is non-blocking subprocesses,
# and will optimally allocate the simulations to available resources
simulation_manager = SimulationManager()

# Indexes of nucleotides to apply forces to and measure distance between (Order Parameter(OP))
com_list = '8,9,10,11,12,13,14,15'
ref_list = '7,6,5,4,3,2,1,0'

# Order of input nucleotides is import to collect the hb_contacts CV correctly, irrelevant for all else
com_list = ','.join(sorted(com_list.split(','), key=int)[::-1])
ref_list = ','.join(sorted(ref_list.split(','), key=int))

# Minimum and maximum distance between nucleotides the simulations will attempt to pull the OP 
xmin = 0
xmax = 15

# Number of simulation windows/replicas
n_windows = 50

# Stiffness of the COM harmonic bias potential
stiff = 3

# Temperature of the simulation
temperature = "40C"

# Starting_r0 is only relevant if you run the pre_equlibration step,
# needed if you have large stiffness values and range of the OP is large but dosen't hurt to always run quickly
# Not possible to know before you run the simulation but you can run a till they print once,
# terminate and paste the starting_r0 value from the output file
starting_r0 = 0.4213

# Frequecny of printing the CVs to file
print_every = 1e4

# Name of file to save all but one of the CVs to, it is possible this cannot be changed
obs_filename = 'all_observables.txt'

# Name of file to save the hb_contacts CV to
hb_contact_filename = 'hb_contacts.txt'

# Number of simulation steps to run for each window
pre_eq_steps = 1e6  # This only need to short
eq_steps = 5e6  # This needs to be long enough to equilibrate the system
prod_steps = 2e8 # This needs to be long enough to get converged free energy profiles (methods to check are provided)

# Setup the custom hb_contacts CV
particle_indexes = [com_list, ref_list]
hb_contact_observable = [{'idx':particle_indexes, 'name':f'{hb_contact_filename}', 'print_every':int(print_every)}]

# oxDNA Simulation parameters
pre_equlibration_parameters = {
    'backend':'CPU', 'steps':f'{pre_eq_steps}','print_energy_every': f'{pre_eq_steps // 10}',
    'print_conf_interval':f'{pre_eq_steps // 2}', "CUDA_list": "no",'use_edge': 'false',
    'refresh_vel': '1','fix_diffusion': '0', 'T':f'{temperature}'}

equlibration_parameters = {
    'backend':'CPU','steps':f'{eq_steps}','print_energy_every': f'{eq_steps// 10}',
    'print_conf_interval':f'{eq_steps // 2}', "CUDA_list": "no",'use_edge': 'false',
    'refresh_vel': '1', 'fix_diffusion': '0', 'T':f'{temperature}'}

production_parameters = {
    'backend':'CPU', 'steps':f'{prod_steps}','print_energy_every': f'{prod_steps}',
    'print_conf_interval':f'{prod_steps}', "CUDA_list": "no", 'use_edge': 'false',
    'refresh_vel': '1','fix_diffusion': '0', 'T':f'{temperature}'}


# This will build the pre_equlibration runs and queue them in the simulation manager

us.build_pre_equlibration_runs(
    simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax,
    pre_equlibration_parameters, starting_r0, pre_eq_steps, continue_run=False,
    # If you want to continue a previous simulation set continue_run=int(n_steps)
    
    print_every=print_every, observable=True, protein=None, sequence_dependant=True,
    force_file=False, name=obs_filename, custom_observable=hb_contact_observable)

simulation_manager.worker_manager(cpu_run=True)

us.build_equlibration_runs(
    simulation_manager, n_windows, com_list, ref_list,
    stiff, xmin, xmax, equlibration_parameters, continue_run=False,
    
    print_every=print_every, observable=True, protein=None, name=obs_filename,
    force_file=False, custom_observable=hb_contact_observable)

simulation_manager.worker_manager(cpu_run=True)

us.build_production_runs(
    simulation_manager, n_windows, com_list, ref_list,
    stiff, xmin, xmax, production_parameters, continue_run=False,
    
    observable=True, print_every=print_every ,protein=None, name=obs_filename,
    force_file=False, custom_observable=hb_contact_observable)

simulation_manager.worker_manager(cpu_run=True)


# You will need to reinitalize the observables if you did not build the dirs in your current kernel, I want to fix this
us.observables_list = []
us.initialize_observables(com_list, ref_list, print_every=print_every, name=obs_filename)


us.pymbar.run_mbar_fes(
    reread_files=True,
    sim_type='prod',
    restraints=False,
    force_energy_split=True)

max_hb = 8
temp_range = np.array([42, 52, 62])

free_n_hb, bin_n_hb, dfree_n_hb = us.pymbar.fes_hist('hb_list', temp_range=temp_range)
free_i_com, bin_com, dfree_com = us.pymbar.fes_hist('com_distance' ,n_bins=20, temp_range=temp_range)
free_contact, bin_contact, dfree_contact = us.pymbar.fes_hist('hb_contacts', n_bins=20, temp_range=temp_range)

p_val = 0.01
temp_range = np.arange(40, 80, 1)
n_bins = 32
convergence_splits = 100
subsample = 100
max_hb = 32
  
hbs_info, coms_info, contacts_info = us.pymbar.pymbar_convergence_free_energy_curves(
    convergence_splits,
    subsampling=subsample,
    max_hb=max_hb,
    n_bins=n_bins,
    temp_range=temp_range,
    restraints=True,
    force_energy_split=True,
    save=True
)