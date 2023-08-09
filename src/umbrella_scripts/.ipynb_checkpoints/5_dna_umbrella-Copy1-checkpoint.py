from umbrella_sampling import ComUmbrellaSampling, CustomObservableUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable, Force
import os
import matplotlib.pyplot as plt
import queue

#set the abspath to project name
path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples/square_block_lattice')

#set the system names
system_name = ['5nm_skew']

#list comp the abspath to the oxdna file locations for each umbrella sim
file_dirs = [f'{path}/{sys}' for sys in system_name] * 2

#new name of folder indicating umbrella run
systems = [f'5nm_skew_mid_mid_dna',  '5nm_skew_mid_top_dna'] 

#the nucleotide index for umbrella potential
com_lists = [
    '9471,9472,9473,9474,9475,9476,9477,9478,9479,9480,22563,22562,22561,22560,22559,22558,22557,22556,22555,22554',
    '9471,9472,9473,9474,9475,9476,9477,9478,9479,9480,22563,22562,22561,22560,22559,22558,22557,22556,22555,22554'    
]
    
    
    
ref_lists = [
    '9419,9420,9421,9422,9423,9424,9425,9426,9427,9428,22017,22016,22015,22014,22013,22011,22012,22010,22009,22008',
    '3858,10515,10516,10517,10518,10519,10520,10521,10522,10523,10524,22485,22484,22483,22482,22481,22480,22479,22478,22477,22476',       
]
             
#umbrella parameters
stiff = 0.1
xmin = 0
xmax = 25
n_windows = 23

#simulation parameters
equlibration_parameters = {'dt':f'0.002', 'steps':'1e6','print_energy_every': '5e5','print_conf_interval':'5e5', 'fix_diffusion':'false'}
production_parameters = {'dt':f'0.002', 'steps':'4e7','print_energy_every': '2e7','print_conf_interval':'2e7', 'fix_diffusion':'false'}

#initalize center of mass umbrella sampling object
us_list = [ComUmbrellaSampling(file_dir, sys) for file_dir, sys in zip(file_dirs,systems)]

#initalize simulation manager
simulation_manager = SimulationManager()

# # build the equlibration simulation by iterating over the umbrella systems
# for us, com_list, ref_list in zip(us_list, com_lists, ref_lists):
#     us.build_equlibration_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, equlibration_parameters, observable=True, print_every=1e3, protein=True, force_file=True)

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