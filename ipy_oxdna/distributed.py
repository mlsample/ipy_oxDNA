import numpy as np
from ipy_oxdna.oxdna_simulation import SimulationManager
import pickle
import argparse
import os
#TODO: Think of a way to make my umbrella sampling and generate replica distributed.


# I want to make it so that I can run a single file and it will be able to run
# multisystem replicas for me distributed across multiple nodes. The first question is what do I alreay have
# Currently I have a single run.py file that can run replicas distributaed across a single node.
# Currently I sbatch a single batch file on my slurm cluster and it runs the replicas for me.
# The first question I need to awnser is should I use sbatch to distribute or should I run a script on login node
# with a python script that will distribute the replicas. I think I should use the login node to distribute the replicas


# This means that I need a python script that will have parameters:
# 1. The number of gpus I want to use
# 2. the number of cpus per gpu
# 3. The number of GPUs per sbatch job


# The starting point always has to be the sim_list
# I will run a python file that will create run.py files?
# Trying to think of this in reverse...
# At the end of the day I will need my allocation to be running a exacuatable.
# This exacutable will be a python code. For now let us assume this will be a python script

# The minal python script that takes in a sim_list and runs it would ideally already be build, and it would then just need to be queue
# This would give some generality

# That would me I would have a python script with a function that builds the sim_list and the sims.
# That python function would return the sim_list

# I would then have a premade python function that would take in a sim_list and run it, it can have hyperparameters for the worker_manager
# the batch script will run a single function that will have all the informaiton it needs to run the simulations
# the exacuateble would be like


# The question is where will it get sim_list_slice from?
# On the highest level end, I have a script that has a function to make the sim_lists, build all the sims,
# create a list of sim_list_slices, create the sbatch files, create the rank_n.py files, and then run the sbatch files
# How can I make this as general as possible?

# I have a directory where the main python script
# I can save each sim_list_slice as a pickle file in the rank subdirectory


def run_sim_list_slice(sim_list_slice):
    sim_manager = SimulationManager()
    
    for sim in sim_list_slice:
        sim_manager.queue_sim(sim, continue_run=False)
    sim_manager.worker_manager(gpu_mem_block=False)


def cli_parser(prog="distributed.py"):
    # A standard way to create and parse command line arguments.
    parser = argparse.ArgumentParser(prog = prog, description="Distributes a sim_list across an HPC.")
    parser.add_argument('-c', '--n_cpus_per_gpu', metavar='n_cpus_per_gpu', nargs=1, type=int, dest='n_cpus_per_gpu', help="The number of cpus per gpu")
    parser.add_argument('-g', '--n_gpus_per_sbatch', metavar='n_gpus_per_sbatch', nargs=1, type=int, dest='n_gpus_per_sbatch', help='The number of gpus per sbatch job')
    parser.add_argument('-d', '--distributed_directory', metavar='distributed_directory', nargs=1, type=str, dest='distributed_directory', help='The directory where the distributed files will be saved')
    return parser


def distribute_sim_list_across_nodes(sim_list):
    parser = cli_parser()
    args = parser.parse_args()
    
    n_cpus_per_gpu = args.n_cpus_per_gpu[0] if args.n_cpus_per_gpu else 1
    n_gpus_per_sbatch = args.n_gpus_per_sbatch[0] if args.n_gpus_per_sbatch else 1
    distributed_directory = args.distributed_directory[0] if args.distributed_directory else f'{os.getcwd()}/distributed'
    distributed_directory = os.path.abspath(distributed_directory)
    sim_list_slices = create_sim_slices(sim_list, n_cpus_per_gpu, n_gpus_per_sbatch)
    
    build_distributed_files(distributed_directory, sim_list_slices, n_cpus_per_gpu, n_gpus_per_sbatch)
    
    run_distributed_files(distributed_directory, sim_list_slices)
    
    return None
    
    
def create_sim_slices(sim_list, n_cpus_per_gpu, n_gpus_per_sbatch):
    n_sims = len(sim_list)
        
    # The total number of cpus will be:
    cpus_per_sbatch =  n_cpus_per_gpu * n_gpus_per_sbatch

    # The number of sbatch jobs will be equal to:
    n_sbatch_jobs =  np.ceil(n_sims / cpus_per_sbatch)
    
    # The total number of gpus will be:
    n_total_gpus = n_sbatch_jobs * n_gpus_per_sbatch
    
    sim_list_slices = np.array_split(sim_list, n_sbatch_jobs)

    return sim_list_slices

    
def build_distributed_files(distributed_directory, sim_list_slices, n_cpus_per_gpu, n_gpus_per_sbatch):
    
    create_distributed_dirs(distributed_directory, sim_list_slices)
    
    pickle_sim_slices(distributed_directory, sim_list_slices)
    
    create_run_files(distributed_directory, sim_list_slices)
    
    create_sbatch_files(distributed_directory, sim_list_slices, n_cpus_per_gpu, n_gpus_per_sbatch)
    
    return None


def run_distributed_files(distributed_directory, sim_list_slices):
    n_dirs = len(sim_list_slices)
    
    for rank in range(n_dirs):
        os.chdir(f'{distributed_directory}/{rank}_job/')
        os.system(f'sbatch {rank}_sbatch.sh')
    return None


def pickle_sim_slices(distributed_directory, sim_list_slices):
    for job_id, sim_list_slice in enumerate(sim_list_slices):
        save_dir = f'{distributed_directory}/{job_id}_job'
        write_pickle_sim_list(save_dir, sim_list_slice, job_id)
    return None


def create_distributed_dirs(distributed_directory, sim_list_slices):
    n_dirs = len(sim_list_slices)
    
    os.makedirs(distributed_directory, exist_ok=True)
    for rank in range(n_dirs):
        os.makedirs(f'{distributed_directory}/{rank}_job', exist_ok=True)
    
    return None
    
    
def create_run_files(distributed_directory, sim_list_slices):
    n_dirs = len(sim_list_slices)
    
    file_contents = [f"""from ipy_oxdna.distributed import run_sim_list_slice, read_pickle_sim_list

job_id = {job_id}
sim_list_slice = read_pickle_sim_list(job_id)
run_sim_list_slice(sim_list_slice)
    """ for job_id in range(n_dirs)]
    
    for job_id, file_content in enumerate(file_contents):
        with open(f'{distributed_directory}/{job_id}_job/{job_id}_run.py', 'w') as f:
            f.write(file_content)


def create_sbatch_files(distributed_directory, sim_list_slices, n_cpus_per_gpu, n_gpus_per_sbatch):
    n_dirs = len(sim_list_slices)
    n_cpus_per_sbatch = int(n_cpus_per_gpu * n_gpus_per_sbatch)
    
    file_contents = [f"""#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -n {n_cpus_per_sbatch}           # number of cores 
#SBATCH --gpus={n_gpus_per_sbatch} # number of gpus
#SBATCH -t 4-00:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH --job-name="{job_id}_job"
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

module load mamba/latest
module load cuda-11.7.0-gcc-11.2.0
source activate oxdnapy12

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe_$SLURM_TASK_PID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log_$SLURM_TASK_PID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d

python3 {job_id}_run.py

    """ for job_id in range(n_dirs)]
    
    for job_id, file_content in enumerate(file_contents):
        with open(f'{distributed_directory}/{job_id}_job/{job_id}_sbatch.sh', 'w') as f:
            f.write(file_content)
    return None
 
    
def write_pickle_sim_list(save_dir, sim_list_slice, job_id):
    with open(f'{save_dir}/{job_id}_sim_slice.pkl', 'wb') as f:
        pickle.dump(sim_list_slice, f)
    return None


def read_pickle_sim_list(job_id):
    with open(f'{job_id}_sim_slice.pkl', 'rb') as f:
        return pickle.load(f)

