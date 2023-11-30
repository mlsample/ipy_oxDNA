from umbrella_sampling import ComUmbrellaSampling, MeltingUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable
import os
import matplotlib.pyplot as plt
import timeit


def main(): 
    path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples')
    file_dir = f'{path}/ico_melting_centered'
    system = 'us_xmax_10_test'
    
    com_list = '4240,4232,4233,4234,4235,4236,4237,4238,4239'
    ref_list = '12298,12290,12291,12292,12293,12294,12295,12296,12297'
    xmin = 0
    xmax = 10
    n_windows = 95
    
    stiff = 0.8
    
    equlibration_parameters = {'steps':'1e7','print_energy_every': '1e7',
                               'print_conf_interval':'1e7',
                               'fix_diffusion': '0', 'fix_diffusion_every': '1000'
                              }
    
    production_parameters ={'steps':'5e7',
                            'print_energy_every': '5e7','print_conf_interval':'5e7',
                            'fix_diffusion': '0', 'fix_diffusion_every': '1000'
                            }
    
    us = MeltingUmbrellaSampling(file_dir, system)
    simulation_manager = SimulationManager()
    
    # us.build_equlibration_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, equlibration_parameters,
    #                        observable=True, sequence_dependant=True, print_every=1e3, name='com_distance.txt')
    # simulation_manager.worker_manager()
    
    us.build_production_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, production_parameters,
                           observable=True, sequence_dependant=True, print_every=1e3, name='com_distance.txt')
    simulation_manager.worker_manager()
    
    wham_dir = os.path.abspath('/scratch/mlsample/ipy_oxDNA/wham/wham')
    n_bins = '200'
    tol = '1e-7'
    n_boot = '100'
    us.wham_run(wham_dir, xmin, xmax, stiff, n_bins, tol, n_boot)

if __name__ == '__main__':
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    print(f'Umbrella run time: {toc - tic}')
