from umbrella_sampling import ComUmbrellaSampling, CustomObservableUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable, Force
import os
import matplotlib.pyplot as plt

path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples')
file_dir = f'{path}/mold/skew_potential_final_system/'
system = 'new_two_con_face_2_umbrella'

com_list = '134,146,149,150,159,163,303,315,318,319,328,332,472,484,487,488,497,501,641,653,656,657,666,670'
ref_list = '2706,2707,2719,2733,2875,2876,2888,2902,3044,3045,3057,3071,3213,3214,3226,3240'
xmin = 0
xmax = 25
n_windows = 94

stiff = 0.5

equlibration_parameters = {'dt':'0.002','steps':'2e6','print_energy_every': '2e6',
                           'print_conf_interval':'2e6', 'fix_diffusion':'false'
                          }

production_parameters ={'dt':'0.002','steps':'1e7','print_energy_every': '1e7',
                        'print_conf_interval':'1e7', 'fix_diffusion':'false'
                        }

us = ComUmbrellaSampling(file_dir, system)
simulation_manager = SimulationManager()

sim_list = us.production_sims[49:]
for sim in sim_list:
    simulation_manager.queue_sim(sim)

# us.build_equlibration_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, equlibration_parameters, observable=True, sequence_dependant=False, print_every=1e3, name='com_distance.txt', protein=True, force_file=True)

# simulation_manager.worker_manager()

# us.build_production_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, production_parameters, observable=True, sequence_dependant=False, print_every=1e3, name='com_distance.txt', protein=True, force_file=True)

simulation_manager.worker_manager()

wham_dir = os.path.abspath('/scratch/mlsample/ipy_oxDNA/wham/wham')
n_bins = '50'
tol = '1e-7'
n_boot = '3000'
us.wham_run(wham_dir, xmin, xmax, stiff, n_bins, tol, n_boot)