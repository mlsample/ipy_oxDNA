from umbrella_sampling import ComUmbrellaSampling, CustomObservableUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable, Force
import os
import matplotlib.pyplot as plt

sim_manager = SimulationManager()

path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples')
file_dir = f'{path}/square_block_lattice/dna_protein_lattice_skew/equili/0'
replicas = range(28)

sim_dir_list = [f'{file_dir}/../../prod/{rep}' for rep in replicas]

sim_list = [Simulation(file_dir, sim_dir) for sim_dir in sim_dir_list]

sim_parameters = [{'dt':f'0.002','steps':'3.5e7','print_energy_every': '5e5','print_conf_interval':'5e5'} for _ in replicas]

for sim_parameter, sim in zip(sim_parameters, sim_list):
    sim_manager.queue_sim(sim, continue_run=1e8)
    
sim_manager.worker_manager(gpu_mem_block=True)