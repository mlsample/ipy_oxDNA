from umbrella_sampling import ComUmbrellaSampling, CustomObservableUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable, Force
import os
import matplotlib.pyplot as plt

path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples')
file_dir = f'{path}/mold/one_connection_mold_protein'
system = 'new_face_2_umbrella'

com_list = '2,3,15,29,171,172,184,198,340,341,353,367,509,510,522,536'
ref_list = '1486,1498,1501,1502,1511,1515,1655,1667,1670,1671,1680,1684,1824,1836,1839,1840,1849,1853,1993,2005,2008,2009,2018,2022'
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

sim_list = us.production_sims[63:]

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