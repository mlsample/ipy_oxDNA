from umbrella_sampling import ComUmbrellaSampling, CustomObservableUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable, Force
import os
import matplotlib.pyplot as plt
import queue

#set the abspath to project name
path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples/square_block_lattice')

#set the system names
system_name = ['dna_protein_lattice_skew']

#list comp the abspath to the oxdna file locations for each umbrella sim
file_dirs = [f'{path}/{sys}' for sys in system_name] * 2

#new name of folder indicating umbrella run
systems = [f'10nm_skew_mid_mid_dna', '10nm_skew_mid_top_dna'] 

#the nucleotide index for umbrella potential
com_lists = [
    '9567,9568,9569,9570,9571,9572,9573,9574,9575,9576,22183,22182,22181,22180,22179,22178,22177,22176,22175,22174',
    '9567,9568,9569,9570,9571,9572,9573,9574,9575,9576,22183,22182,22181,22180,22179,22178,22177,22176,22175,22174',
]
    
    
    
ref_lists = [
    '11653,11654,11655,11656,11657,11658,11659,11660,11661,11662,22240,22239,22238,22237,22236,22235,22234,22233,22232,22231',    
    '9419,9420,9421,9422,9423,9424,9425,9426,9427,9428,22354,22353,22352,22351,22350,22349,22348,22347,22346,22345'
    
]
             
#umbrella parameters
stiff = 0.1
xmin = 0
xmax = 25
n_windows = 23

#simulation parameters
equlibration_parameters = {'dt':f'0.002', 'steps':'5e5','print_energy_every': '5e5','print_conf_interval':'5e5', 'fix_diffusion':'false'}
production_parameters = {'dt':f'0.002', 'steps':'2e7','print_energy_every': '2e7','print_conf_interval':'2e7', 'fix_diffusion':'false'}

#initalize center of mass umbrella sampling object
us_list = [ComUmbrellaSampling(file_dir, sys) for file_dir, sys in zip(file_dirs,systems)]

#initalize simulation manager
simulation_manager = SimulationManager()

# # build the equlibration simulation by iterating over the umbrella systems
# for us, com_list, ref_list in zip(us_list, com_lists, ref_lists):
#     us.build_equlibration_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, equlibration_parameters, print_every=1e3, observable=True, protein=True, force_file=True)

# #run equlibration
# simulation_manager.worker_manager(gpu_mem_block=False)

#build production
for us, com_list, ref_list in zip(us_list, com_lists, ref_lists):
    us.build_production_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, production_parameters, observable=True, print_every=1e3, name='com_distance.txt', protein=True, force_file=True, continue_run=1e8)

#run
simulation_manager.worker_manager(gpu_mem_block=False)

wham_dir = os.path.abspath('/scratch/mlsample/ipy_oxDNA/wham/wham')
n_bins = '200'
tol = '1e-5'
n_boot = '100000'
for us in us_list:
    us.wham_run(wham_dir, xmin, xmax, stiff, n_bins, tol, n_boot)

