from umbrella_sampling import ComUmbrellaSampling, MeltingUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable
from wham_analysis import collect_coms
import os
import matplotlib.pyplot as plt
from vmmc import *
from collections import Counter
from parameter_search import *
import numpy as np
from skopt import Optimizer
from skopt.space import Categorical, Space

#Define umbrella sampling param space
param_space = Space([
    Categorical([0.5, 1., 5., 10.], name='k_value'),
    Categorical([25., 40., 52., 70.], name='temperature'),
    Categorical([True, False], name='unique_binding'),
    Categorical([10., 100., 1000.], name='print_every'),
    Categorical([5., 10., 15., 20.], name='xmax'),
    Categorical([1e8, 5e8, 4e9], name='production_step'),
    Categorical([20., 50.], name='n_window')
])

inital_parameters = [
 [10.0, 25., False, 10., 5., 500000000.0, 50.],
 [0.5, 25., False, 100., 10., 500000000.0, 50.],
 [10.0, 70., True, 10., 20., 500000000.0, 20.]
]
    
#Define ground truth melting
path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples')
systems = ['duplex_melting']

file_dir_list = [f'{path}/{sys}' for sys in systems]
sim_dir_list = [f'{file_dir}/vmmc_melting_replicas/vmmc_melting_rep' for sys, file_dir in zip(systems, file_dir_list)]

n_replicas = 40
vmmc_replica_generator = VmmcReplicas()

vmmc_replica_generator.multisystem_replica(
    systems,
    n_replicas,
    file_dir_list,
    sim_dir_list
)
vmmc_sim_list = vmmc_replica_generator.sim_list
queue_of_simulations = vmmc_replica_generator.queue_of_sims

vmmc_sim_list[0].analysis.read_vmmc_op_data()
vmmc_sim_list[0].analysis.calculate_sampling_and_probabilities()
vmmc_sim_list[0].analysis.calculate_and_estimate_melting_profiles()
   
#Define system parameters
path = os.path.abspath('/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples/duplex_melting/parameter_scan')
max_hb = 8
com_list = '8,9,10,11,12,13,14,15'
ref_list = '0,1,2,3,4,5,6,7'
vmmc_ground_truth = vmmc_sim_list[0]

#Run
n_iterations = 5
batch_size = 3
run_baysian_hyperparameter_optimization(
    param_space,
    vmmc_ground_truth,
    max_hb, path,
    com_list,
    ref_list,
    n_iterations,
    batch_size,
    resume_state=None,
    inital_parameters=inital_parameters,
    subprocess=False
)