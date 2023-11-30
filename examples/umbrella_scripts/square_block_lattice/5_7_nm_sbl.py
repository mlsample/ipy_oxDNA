from umbrella_sampling import ComUmbrellaSampling, CustomObservableUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable, Force
import os
import matplotlib.pyplot as plt
import queue


sim_manager = SimulationManager()

path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples/square_block_lattice')

systems = ['5nm_skew', '7nm_skew']
replicas = range(23)
file_dir_list = [f'{path}/{sys}/eq_0' for sys in systems]
sim_dir_list = [f'{file_dir}/../{sys}_prod' for sys, file_dir in zip(systems, file_dir_list)]
sim_rep_list = []
for sys in sim_dir_list:
    for rep in replicas:
        sim_rep_list.append(f'{sys}_{rep}')
q1 = queue.Queue()
for sys in sim_rep_list:
    q1.put(sys)
sim_list = []

for file_dir in file_dir_list:
    for _ in range(len(replicas)):
        sim_dir = q1.get()
        sim_list.append(Simulation(file_dir, sim_dir))
q2 = queue.Queue()
for sim in sim_list:
    q2.put(sim)
sim_parameters = [{'dt':f'0.002', 'steps':'1e8','print_energy_every': '5e5','print_conf_interval':'5e5'} for sys in systems]
for parameter in sim_parameters:
    for _ in range(len(replicas)):
        sim = q2.get()
        sim.build(clean_build='force')
        sim.input_file(parameter)
        sim.add_protein_par()
        sim.add_force_file()
        sim_manager.queue_sim(sim)
        
sim_manager.worker_manager(gpu_mem_block=False)