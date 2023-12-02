from skopt.space import Real, Integer, Categorical
from ipy_oxdna.umbrella_sampling import ComUmbrellaSampling, MeltingUmbrellaSampling
from ipy_oxdna.oxdna_simulation import SimulationManager, Simulation, Observable
from ipy_oxdna.wham_analysis import collect_coms
import os
import matplotlib.pyplot as plt
from ipy_oxdna.vmmc import *
from collections import Counter
import logging
import joblib
import multiprocessing as mp
from skopt import Optimizer
from skopt.space import Categorical, Space
from copy import deepcopy


logging.basicConfig(filename='optimization.log', level=logging.INFO)

def run_baysian_hyperparameter_optimization(param_space, vmmc_ground_truth, max_hb, path, com_list, ref_list, n_iterations, batch_size, resume_state=None, inital_parameters=None, subprocess=True):
    
    if subprocess is True:
        spawn(baysian_hyperparameter_optimization, args=(param_space, vmmc_ground_truth, max_hb, path, com_list, ref_list, n_iterations, batch_size), kwargs={'resume_state':resume_state, 'inital_parameters':inital_parameters})
    else:
        baysian_hyperparameter_optimization(param_space, vmmc_ground_truth, max_hb, path, com_list, ref_list, n_iterations, batch_size, resume_state=resume_state, inital_parameters=inital_parameters)
        
def spawn(f, args=(), kwargs={}):
    """Spawn subprocess"""
    p = mp.Process(target=f, args=args, kwargs=kwargs)
    p.start()
        
def baysian_hyperparameter_optimization(param_space, vmmc_ground_truth, max_hb, path, com_list, ref_list,  n_iterations, batch_size, resume_state=None, inital_parameters=None):
    # Instantiate the optimizer
    print(param_space)
    if resume_state is None:
        optimizer = Optimizer(
            dimensions=param_space,
            base_estimator="GP",
            acq_func="EI",
            n_initial_points=10  # Number of random initialization points before Bayesian Optimization
        )
    else:
        try:
            files = [file for file in os.listdir(path) if 'optimizer' in file]
            which_number = sorted([int(file.split('.')[0].split('_')[-1]) for file in files])[-1]
            for file in files:
                split = int(file.split('.')[0].split('_')[-1])
                if which_number == split:
                    best_optim = [file]
                    break
            optim = best_optim[0]
            optimizer = joblib.load(f'{path}/{optim}')
        except:
            return print('Pass valid optimizer file name')
    
    # Custom optimization loop
    
    for i in range(n_iterations):
        # Ask the optimizer to suggest a batch of parameter sets
        if (inital_parameters is not None) and (i == 0): 
            suggested_params_batch = deepcopy(inital_parameters)
            suggested_params_batch_formated = np.array(suggested_params_batch, dtype='object').T.tolist()
        
        else:
            suggested_params_batch = optimizer.ask(n_points=batch_size)
            suggested_params_batch_formated = np.array(suggested_params_batch, dtype='object').T.tolist()
        
        # Evaluate the objective function in parallel (assuming your function can handle batch evaluations)
        losses = objective_function(vmmc_ground_truth, max_hb, path, com_list, ref_list,  *suggested_params_batch_formated)
        
        # Tell the optimizer the result
        print(suggested_params_batch) 
        for params, loss in zip(suggested_params_batch, losses):
            optimizer.tell(params, loss)
    
        # Custom logging or checkpointing code here
        logging.info(f"Iteration {i}")
        logging.info(f"Suggested parameters: {suggested_params_batch}")
        logging.info(f"Losses: {losses}")
        joblib.dump(optimizer, f'optimizer_checkpoint_{i}.pkl')

        # Optionally, save other relevant data
        joblib.dump(suggested_params_batch, f'suggested_params_batch_{i}.pkl')
        joblib.dump(losses, f'losses_{i}.pkl')
        
        print(f"Iteration {i}: Loss = {min(losses)}")
        

def objective_function(vmmc_ground_truth, max_hb, path, com_list, ref_list, k_value_list, temperature_list, unique_binding_list, print_every_list, xmax_list, production_step_list, n_window_list):
    print('run sims')
    #shape: 1 x n_parallized_predications
    us_list = run_umbrella_sampling(path, com_list, ref_list, k_value_list, temperature_list, unique_binding_list, print_every_list, xmax_list, production_step_list, n_window_list)
    
    print('sims done')
    for us in us_list:
        print(hasattr(us, 'vmmc_dir'))
        if hasattr(us, 'vmmc_dir') is False:
            us.continuous_to_discrete_unbiasing(max_hb)
            print('here')
            us.calculate_melting_temperature_using_vmmc()


    y_predict = np.array([1 - us.vmmc_sim.analysis.finfs[0] for us in us_list])
    print(f'{temperature_list[0]=}', f'{vmmc_ground_truth.analysis.popt=}')
    y_true = vmmc_ground_truth.analysis.sigmoid(float(temperature_list[0]), *vmmc_ground_truth.analysis.popt)
    
    #shape: 1 x n_parallized_predications
    loss = (y_true - y_predict)**2
    print(loss)
    return loss


# def continuo

def run_umbrella_sampling(path, com_list, ref_list, k_value_list, temperature_list, unique_binding_list, print_every_list, xmax_list, production_step_list, n_window_list):
    
    n_window_list = [int(n_window) for n_window in n_window_list]
    # Initialize lists to store system names and file directories
    systems = []
    file_dirs = []
    
    
    inital_parameters = np.array([[10.0, 25., False, 10., 5., 500000000.0, 50.],
     [0.5, 25., False, 100., 10., 500000000.0, 50.],
     [10.0, 70., True, 10., 20., 500000000.0, 20.]
    ], dtype='object').tolist()
    params = np.array([k_value_list, temperature_list, unique_binding_list, print_every_list, xmax_list, production_step_list, n_window_list], dtype='object').T
    if (params == inital_parameters).all():
        print('pass')     
        #here I need to reaasign the value of all of the varaible to the int version of the variables
        # Convert all the variables to their integer versions
        # k_value_list = list(map(int, k_value_list))
        temperature_list = list(map(int, temperature_list))
        # inital_parameters
        print_every_list = list(map(int, print_every_list))
        xmax_list = list(map(int, xmax_list))
        # production_step_list = list(map(int, production_step_list))
        n_window_list = list(map(int, n_window_list))
    else:
        print('fail')

    # Create system names and file directories based on parameter combinations
    for k_value, temperature, unique_binding, print_every, xmax, production_step, n_window in zip(k_value_list, temperature_list, unique_binding_list, print_every_list, xmax_list, production_step_list, n_window_list):
        systems.append(f'{k_value}_{temperature}_{unique_binding}_{print_every}_{xmax}_{production_step}_{n_window}')
        file_dirs.append(f'{path}')   

    xmin = 0
    starting_r0 = 1
    steps = 1e6
    
    pre_equlibration_parameters_list = [
        {
            'backend': 'CPU',
            'steps': '1e6',
            'print_energy_every': '1e6',
            'print_conf_interval': '1e6',
            'CUDA_list': 'no',
            'use_edge': 'false',
            'refresh_vel': '1',
            'fix_diffusion': '0',
            'fix_diffusion_every': '1000',
            'T': f'{temp}C'
        } 
        for temp in temperature_list
    ]
    
    equlibration_parameters_list = [
        {
            'backend': 'CPU',
            'steps': '1e7',
            'print_energy_every': str(print_every),
            'print_conf_interval': '4e7',
            'CUDA_list': 'no',
            'use_edge': 'false',
            'refresh_vel': '1',
            'fix_diffusion': '0',
            'fix_diffusion_every': '1000',
            'T': f'{temp}C'
        } 
        for temp in temperature_list
    ]
    
    production_parameters_list = [
        {
            'backend': 'CPU',
            'steps': str(production_step),
            'print_energy_every': '1e8',
            'print_conf_interval': '1e8',
            'CUDA_list': 'no',
            'use_edge': 'false',
            'refresh_vel': '1',
            'fix_diffusion': '0',
            'fix_diffusion_every': '1000',
            'T': f'{temp}C'
        } 
        for temp, production_step in zip(temperature_list, production_step_list)
    ]
    
    us_list = [MeltingUmbrellaSampling(file_dir, sys, clean_build='force') for file_dir, sys in zip(file_dirs,systems)]
    
    simulation_manager = SimulationManager()
    for us in us_list:
        if hasattr(us, 'free'):
            skip_sim_run = True
            break
        else:
            skip_sim_run = False
    
    if skip_sim_run is True:
        pass
    elif skip_sim_run is False:
        for us, n_windows, stiff, xmax, pre_equlibration_parameters, print_every in zip(us_list, n_window_list, k_value_list, xmax_list, pre_equlibration_parameters_list, print_every_list):
            print(us.system)
            us.build_pre_equlibration_runs(simulation_manager, n_windows, com_list, ref_list,
                                   stiff, xmin, xmax, pre_equlibration_parameters, starting_r0, steps,
                                   print_every=print_every, observable=True, protein=None,
                                   force_file=None, continue_run=False)
        
        simulation_manager.worker_manager(cpu_run=True, gpu_mem_block=False)
        
        for us, n_windows, stiff, xmax, equlibration_parameters, print_every in zip(us_list, n_window_list, k_value_list, xmax_list, equlibration_parameters_list, print_every_list):
            print(us.system)
            us.build_equlibration_runs(simulation_manager, n_windows, com_list, ref_list,
                                   stiff, xmin, xmax, equlibration_parameters,
                                   print_every=print_every, observable=True, protein=None,
                                   force_file=None, continue_run=False)
            
        for us, unique_binding in zip(us_list, unique_binding_list):
            if unique_binding:
                us.modify_topology_for_unique_pairing()
        
        simulation_manager.worker_manager(cpu_run=True, gpu_mem_block=False)
    
        for us, n_windows, stiff, xmax, production_parameters, print_every in zip(us_list, n_window_list, k_value_list, xmax_list, production_parameters_list, print_every_list):
            print(us.system)
            us.build_production_runs(simulation_manager, n_windows, com_list, ref_list,
                                 stiff, xmin, xmax, production_parameters,
                                 observable=True, print_every=print_every, protein=None,
                                 force_file=None, continue_run=False)
        
        simulation_manager.worker_manager(cpu_run=True, gpu_mem_block=False)
        
        
        wham_dir = os.path.abspath('/scratch/mlsample/ipy_oxDNA/wham/wham')
        n_bins = '200'
        tol = '1e-5'
        n_boot = '0'
        for us, xmax, stiff in zip(us_list, xmax_list, k_value_list):
            us.wham_run(wham_dir, xmin, xmax, stiff, n_bins, tol, n_boot)
        
    return us_list
