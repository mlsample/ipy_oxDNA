from ipy_oxdna.umbrella_sampling import ComUmbrellaSampling, CustomObservableUmbrellaSampling
from ipy_oxdna.oxdna_simulation import SimulationManager, Simulation, Observable, Force
import os
import matplotlib.pyplot as plt

sim_manager = SimulationManager()

path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples')
file_dir = f'{path}/rod_sheet'

replicas = ['0.1', '0.1', '0.1', '0.5', '0.5', '0.5', '1.0', '1.0', '1.0', '2.0', '2.0', '2.0']
sim_dir_list = [f'{file_dir}/{idx}_{rep}' for idx, rep in enumerate(replicas)]

sim_list = [Simulation(file_dir, sim_dir) for sim_dir in sim_dir_list]

sim_parameters = [{'salt_concentration':f'{mol}', 'steps':'1e6','print_energy_every': '5e5','print_conf_interval':'5e5'} for mol in replicas]
for sim_parameter, sim in zip(sim_parameters, sim_list):
    sim.build(clean_build='force')
    sim.input_file(sim_parameter)
    sim_manager.queue_sim(sim)

sim_manager.run()
