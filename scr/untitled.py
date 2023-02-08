from umbrella_sampling import ComUmbrellaSampling, MeltingUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable
import os
import matplotlib.pyplot as plt

def main():
    path = os.path.abspath('../ipy_oxdna_examples/melting_example/')
    file_dir = f'{path}/duplex_8nt/'
    system = 'us_k_05_min_0_max_5_1e3'
    
    com_list = '8,9,10,11,12,13,14,15'
    ref_list = '0,1,2,3,4,5,6,7'
    xmin = 0
    xmax = 5
    n_windows = 48
    
    stiff = 0.5
    
    equlibration_parameters = {'steps':'1e7', 'T':'20C', 'print_energy_every': '5e5','print_conf_interval':'5e5',
                               'max_density_multiplier': '50'}
    
    production_parameters = {'steps':'1e8', 'T':'20C', 'print_energy_every': '5e6',
                             'print_conf_interval':'5e5', 'max_density_multiplier':'50'}
    
    us = MeltingUmbrellaSampling(file_dir, system)
    simulation_manager = SimulationManager()
    
    us.build_production_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, production_parameters,
                           observable=True, print_every=1e3, name='com_distance.txt')
    
    simulation_manager.run()
    
    
    
if __name__ == '__main__':
    main()