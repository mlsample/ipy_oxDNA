from umbrella_sampling import ComUmbrellaSampling, MeltingUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable
import os
import matplotlib.pyplot as plt

def main():

    
    from umbrella_sampling import ComUmbrellaSampling, MeltingUmbrellaSampling
    from oxdna_simulation import SimulationManager, Simulation, Observable
    import os
        
    path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples')
    file_dir = f'{path}/duplex_melting'
    system = 'us_melting'
    
    com_list = '8,9,10,11,12,13,14,15'
    ref_list = '0,1,2,3,4,5,6,7'
    xmin = 0
    xmax = 8
    n_windows = 95
    
    stiff = 0.8
    
    equlibration_parameters = {'steps':'5e4','print_energy_every': '5e6',
                               'print_conf_interval':'5e6', "CUDA_list": "no",
                               'use_edge': 'false', 'refresh_vel': '1',
                               'fix_diffusion': '0', 'fix_diffusion_every': '1000'
                              }
    
    production_parameters ={'steps':'2e5','print_energy_every': '2e7',
                            'print_conf_interval':'2e7', "CUDA_list": "no",
                            'use_edge': 'false', 'refresh_vel': '1',
                            'fix_diffusion': '0', 'fix_diffusion_every': '1000'
                            }
    
    us = MeltingUmbrellaSampling(file_dir, system)
    simulation_manager = SimulationManager()
    
    us.build_equlibration_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, equlibration_parameters,
                           observable=True, print_every=1e3, name='com_distance.txt')
    simulation_manager.worker_manager()
    
    us.build_production_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, production_parameters,
                           observable=True, print_every=1e3, name='com_distance.txt')  
    simulation_manager.worker_manager()
    
    wham_dir = os.path.abspath('/scratch/mlsample/ipy_oxDNA/wham/wham')
    n_bins = '200'
    tol = '1e-5'
    n_boot = '30'
    us.wham_run(wham_dir, xmin, xmax, stiff, n_bins, tol, n_boot)

if __name__ == '__main__':
    main()