import oxpy
import os
import numpy as np
import shutil
from oxdna_json_utils import *
from oxdna_writing_utils import *


def sep_conf(last_conf, traj, sim_dir):
    """
    Separate the trajectory file into separate files for each umbrella window.
    :param last_conf: path to last configuration file of pulling simulations
    :param traj: path to trajectory file of pulling simulations
    :param sim_dir: directory to save the separated files
    :return: None
    """
    # Create base umbrella simulation directory if it does not exist
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)
    # get number of lines in a singe oxDNA configuration
    num_lines = sum(1 for line in open(last_conf))
    with open(traj, 'r') as f:
        # read trajectory file
        file = f.readlines()
        # get the number of configurations in the trajectory file
        n_confs = len(file) // num_lines
        for i in range(n_confs):
            # create a new window(dir) for each configuration
            c_dir = os.path.join(sim_dir, str(i))
            if not os.path.exists(c_dir):
                os.mkdir(c_dir)
            # Write the configuration to the new window
            with open(os.path.join(c_dir, f'conf_{i}.dat'), 'w') as f:
                for j in range(num_lines):
                    f.write(file[i * num_lines + j])
    return n_confs


def get_input(**input_dict):
    """
    Create input file object for umbrella simulation using the input_dict dictionary as input file Parameters in oxpy context.
    :param input_dict: dictionary of input parameters
    :return: my_input: input file as oxpy input object
    """
    with oxpy.Context():
        my_input = oxpy.InputFile()
        # assign input parameters to input object
        for k, v in input_dict.items():
            if v:
                my_input[k] = v
    return my_input


def get_windows_per_gpu(n_gpus, n_confs):
    """
    Calculate the optimal number of windows per gpu
    :param n_gpus:
    :return:
    """
    round = n_confs // n_gpus
    remainder = n_confs % n_gpus
    w_p_gpu = []
    for i in range(n_gpus):
        if remainder != 0:
            w_p_gpu.append(round + 1)
            remainder -= 1
        else:
            w_p_gpu.append(round)
    w_p_gpu.sort()
    return w_p_gpu


def write_production_run_file(run_file, windows_per_gpu, sim_dir):
    # Write hpc run file to each window
    with open(run_file, 'r') as f:
        lines = f.readlines()
        window = 0
        for gpu_set in windows_per_gpu:
            for file in range(gpu_set):
                print(str(window))
                with open(os.path.join(sim_dir, str(window), 'run.sh'), 'w') as r:
                    win = window + 1
                    for line in lines:
                        if 'job-name' in line:
                            r.write(f'#SBATCH --job-name="{window}"\n')
                        else:
                            r.write(line)
                    if file < (gpu_set - 1):
                        r.write(f'cd {os.path.join(sim_dir, str(win))}\nsbatch run.sh')
                window += 1
    return None




def write_production(file_dir=None, sim_dir=None, p1=None, p2=None, xmin=None, xmax=None, all_forces=None, steps_per_window=None, n_gpus=None, input_json_dict=None, run_file=None, restart_from_last_conf=None):
    """
    Write production files for umbrella simulations.
    :param file_dir: location of pulling simulations
    :param sim_dir: location of umbrella simulation directories
    :param xmin: minimum x value of umbrella order parameter
    :param xmax: maximum x value of umbrella order parameter
    :param p1: first umbrella order parameter
    :param p2: second umbrella order parameter
    :param k: spring constant of umbrella
    :param steps: number of steps in umbrella simulation
    :param run_file: hpc run file for umbrella simulation
    :return: None
    """
    print('Writing production files...')
    # get name of pulling simulation files
    # last_conf = os.path.join(file_dir, 'last_conf.dat')
    # top = [t for t in os.listdir(file_dir) if t.endswith('.top')][0]
    # op_file = os.path.join(file_dir, 'op.txt')
    seq_dep = 'oxDNA2_sequence_dependent_parameters.txt'
    # force_type = "com"
    # force_rate = "0"
    # if restart_from_last_conf not None:
    #     last_conf, top = get_last_conf_top(file_dir)
    #     input_json_dict["conf_file"] = f"last_conf.dat"
    #     input_json_dict["topology"] = top
    #     write_input(os.path.join(sim_dir, window), input_json_dict, production=True)

    last_conf, top = get_last_conf_top(file_dir)
    traj = os.path.join(file_dir, 'trajectory.dat')

    # Create umbrella window dirs and separate the pulling confs into umbrella windows
    n_confs = sep_conf(os.path.join(file_dir, 'last_conf.dat'), traj, sim_dir)
    x_range = np.linspace(xmin, xmax, (n_confs + 1))[1:]

    #Get the optimal number of windows for each gpu to run
    w_p_gpu = get_windows_per_gpu(n_gpus, n_confs)
    
    for window, r0 in enumerate(x_range):
        window = str(window)

        # Write Topology files to each windows
        shutil.copyfile(os.path.join(file_dir, top), os.path.join(sim_dir, window, top))

        # Write order parameter files to each windows
        #write_op_file(os.path.join(sim_dir, window), p1, p2)

        # Write sequence dependent parameter files to each windows
        shutil.copy(os.path.join(file_dir, 'oxDNA2_sequence_dependent_parameters.txt'), os.path.join(sim_dir, window))

        # Write observables file to each windows
        write_observables_file(os.path.join(sim_dir, window), p1, p2)

        # Write force file to each windows
        # TO DO: Fix force_json
        for f_number, force in enumerate(all_forces):
            if force['external_force'] == 'trap':
                mutual_trap_json(external_force=force['external_force'],
                                 particle=force['particle'],
                                 pos0=force['pos0'],
                                 k=force['k'],
                                 pbc=force['pbc'],
                                 rate=force['rate'],
                                 dir=force['dir'],
                                 f_number=f_number,
                                 write_path=os.path.join(sim_dir, window, f'forces.json')
                                 )
            elif force['external_force'] == 'com':
                com_force_json(external_force=force['external_force'],
                               p1=force['p1'],
                               p2=force['p2'],
                               k=force['k'],
                               r0=str(r0),
                               pbc=force['pbc'],
                               rate=0,
                               f_number=f_number,
                               write_path=os.path.join(sim_dir, window, f'forces.json')
                               )

        input_json_dict["conf_file"] = f"conf_{window}.dat"
        input_json_dict["topology"] = top
        write_input(os.path.join(sim_dir, window), input_json_dict, production=True)

        # Write hpc run file to each window
        if run_file is not None:
            write_production_run_file(run_file, w_p_gpu, sim_dir)
    print(f'{n_confs} umbrella windows created')
    return None


def run_production_slurm_files(n_gpus, n_windows, sim_dir):
    windows_per_gpu = get_windows_per_gpu(n_gpus, n_windows)
    run_dirs = [0]
    for gpu_set in windows_per_gpu:
        run_dirs.append(run_dirs[-1] + gpu_set)
    run_dirs.pop()
    for i in run_dirs:
        os.chdir(os.path.join(f"{sim_dir}", str(i)))
        os.system("sbatch run.sh")
        
