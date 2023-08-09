from umbrella_sampling import ComUmbrellaSampling, MeltingUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable
import os
import matplotlib.pyplot as plt
import timeit


def main():
    path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples')
    file_dir = f'{path}/ico_melting'
    
    replicas = list(range(3))
    sim_dir_list = [f'{file_dir}/replica_{replica}' for replica in replicas]
    
    sim_list = [Simulation(file_dir, sim_dir) for sim_dir in sim_dir_list]
    simulation_manager = SimulationManager() 
    
    for sim in sim_list:
        sim.build(clean_build='force')
        simulation_manager.queue_sim(sim)
        sim.sequence_dependant()
    simulation_manager.worker_manager()

if __name__ == '__main__':
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    print(f'Replicas run time: {toc - tic}')
