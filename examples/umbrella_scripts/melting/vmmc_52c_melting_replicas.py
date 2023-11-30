from oxdna_simulation import GenerateReplicas
from vmmc import *
import os
import scienceplots


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

p1 = '15,14,13,12,11,10,9,8'
p2 = '0,1,2,3,4,5,6,7'
pre_defined_weights = [8, 16204, 1882.94, 359.746, 52.5898, 15.0591, 7.21252, 2.2498, 2.89783]

sim_parameters = {'T':'52C', 'steps':'1e9','print_energy_every': '1e5','print_conf_interval':'1e6'}

# for _ in range(n_replicas):
#     vmmc = queue_of_simulations.get()
#     vmmc.build(p1, p2, pre_defined_weights=pre_defined_weights, clean_build='force')
#     vmmc.input_file(sim_parameters)
#     vmmc.build_com_hb_observable(p1, p2)

for vmmc in vmmc_sim_list[1:]:
    vmmc.oxpy_run.run(continue_run=1e9)
    
vmmc_sim_list[0].oxpy_run.run(continue_run=3e9, join=True)

print('run_complete')