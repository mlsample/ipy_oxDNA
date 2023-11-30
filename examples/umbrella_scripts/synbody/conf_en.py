from oxdna_simulation import SimulationManager, Simulation, GenerateReplicas
import os
import matplotlib.pyplot as plt

sim_manager = SimulationManager()

path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples/synbody/conformational_entropy')

systems = sorted(os.listdir(path))
systems = [item for item in systems if not item.startswith('.')]

file_dir_list = [f'{path}/{sys}/eq_0' for sys in systems]
sim_dir_list = [f'{file_dir}/../prod' for sys, file_dir in zip(systems, file_dir_list)]

n_replicas = 6
replica_generator = GenerateReplicas()

replica_generator.multisystem_replica(
    systems,
    n_replicas,
    file_dir_list,
    sim_dir_list
)

sim_list = replica_generator.sim_list
queue_of_simulations = replica_generator.queue_of_sims

sim_parameters = [{'dt':f'0.002', 'steps':'1e9','print_energy_every': '1e5','print_conf_interval':'1e5'} for sys in systems]
for parameter in sim_parameters:
    for _ in range(n_replicas):
        sim = queue_of_simulations.get()
        # sim.build(clean_build='force')
        # sim.input_file(parameter)
        # sim.add_protein_par()
        # sim.add_force_file()
        sim_manager.queue_sim(sim, continue_run=1e9)
sim_manager.worker_manager(gpu_mem_block=False)