from rewrite.oxdna_simulation import Simulation, Force, Observable, SimulationManager
from rewrite.wham_analysis import wham_analysis
import multiprocessing as mp
import os
from os.path import join, exists
import numpy as np
import shutil



"""
I want to be able to continue the runs I started last night
In order to continue the runs I need to know which runs are finished and which runs are not done
I also want to be able to coninue runs, so I would bo good it I make a set of functions able to do both
I might be able to see if the number of steps run equals the number of steps in the input file
I might be able to bee if the number of com_distances is equal to number of steps / print_conf_every
Once I know which windows are done and which are not, I cn add the window number of those not done to a list
and append simulations to a not_done_simulations list
I can then just run those fresh

Alternativly I can find which windows are not done and then "continue the runs",
I should keep the continue run and finish runs idea seperate

To continue runs I want to run for another 
"""

class BaseUmbrellaSampling:
    def __init__(self, file_dir, system):
        """
        Instance file location and system name.
        
        Parameters:
            file_dir (str): absolute path to inital dat and top files
            system (str): name of system
        """
        self.system = system
        self.file_dir = file_dir
        self.system_dir = join(self.file_dir, self.system)
        if not exists(self.system_dir):
            os.mkdir(self.system_dir)  
        # self.read_progress()
    

    def equlibration_windows(self, n_windows):
        self.equlibration_sim_dir = join(self.system_dir, 'equlibration')
        if not exists(self.equlibration_sim_dir):
            os.mkdir(self.equlibration_sim_dir)
        self.equlibration_sims = [Simulation(self.file_dir,join(self.equlibration_sim_dir, str(window))) for window in range(n_windows)]
    

    def equlibration_runs(self, com_list, ref_list, stiff, xmin, xmax, n_windows, input_parameters, continue_run=False):
        self.equlibration_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        self.com_distance_observable(com_list, ref_list)
        
        manager = SimulationManager()
        for sim, f_1, f_2 in zip(self.equlibration_sims, self.umbrella_forces_1, self.umbrella_forces_2):
            if continue_run == False:
                sim.build(clean_build='force')
                sim.add_force(f_1)
                sim.add_force(f_2)
                #sim.add_observable(self.com_observable)
            sim.input_file(input_parameters)
            manager.queue_sim(sim, continue_run=continue_run)
        manager.worker_manager()
                                             
                                                 
    def production_windows(self, n_windows):
            self.production_sim_dir = join(self.system_dir, 'production')
            if not exists(self.production_sim_dir):
                os.mkdir(self.production_sim_dir)
            self.production_window_dirs = [join(self.production_sim_dir, str(window)) for window in range(n_windows)]
            self.production_sims = []
            for s, window_dir, window in zip(self.equlibration_sims, self.production_window_dirs, range(n_windows)):
                self.production_sims.append(Simulation(self.equlibration_sims[window].sim_dir, str(window_dir)))
   
    def production_runs(self, com_list, ref_list, stiff, xmin, xmax, n_windows, input_parameters, continue_run=False):
        self.equlibration_windows(n_windows)
        self.production_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        self.com_distance_observable(com_list, ref_list)
        manager = SimulationManager()
        for sim, f_1, f_2 in zip(self.production_sims, self.umbrella_forces_1, self.umbrella_forces_2):
            if continue_run == False:
                sim.build(clean_build='force')
                sim.add_force(f_1)
                sim.add_force(f_2)
                sim.add_observable(self.com_observable)
            sim.input_file(input_parameters)
            manager.queue_sim(sim, continue_run=continue_run)
        manager.worker_manager()
     
    def run_wham(self, wham_dir, umbrella_stiff, n_bins, tol, n_boot):
        self.com_dir = join(self.production_sim_dir, 'com_dir')
        pre_temp = self.production_sims[0].input.input['T']
        if ('C'.upper() in pre_temp) or ('C'.lower() in pre_temp):
            self.temperature = (float(pre_temp[:-1]) + 273.15) / 3000
        elif ('K'.upper() in pre_temp) or ('K'.lower() in pre_temp):
             self.temperature = float(pre_temp[:-1]) / 3000
        wham_analysis(wham_dir, self.production_sim_dir, self.com_dir, str(self.xmin), str(self.xmax), str(umbrella_stiff), str(n_bins), str(tol), str(n_boot), str(self.temperature))
     
    def umbrella_forces(self, com_list, ref_list, stiff, xmin, xmax, n_windows):
        f = Force()
        x_range = np.linspace(xmin, xmax, (n_windows + 1))[1:]

        self.umbrella_forces_1 = []
        self.umbrella_forces_2 = []
        
        for x_val in x_range:   
            self.umbrella_force_1 = f.com_force(
                com_list=com_list,                        
                ref_list=ref_list,                        
                stiff=f'{stiff}',                    
                r0=f'{x_val}',                       
                PBC='1',                         
                rate='0',
            )        
            self.umbrella_forces_1.append(self.umbrella_force_1)
            
            self.umbrella_force_2= f.com_force(
                com_list=ref_list,                        
                ref_list=com_list,                        
                stiff=f'{stiff}',                    
                r0=f'{x_val}',                       
                PBC='1',                          
                rate='0',
            )
            self.umbrella_forces_2.append(self.umbrella_force_2)
            
    def com_distance_observable(self, com_list, ref_list,  print_every=1e4, name='com_distance.txt'):
        obs = Observable()
        self.com_observable = obs.distance(
            particle_1=com_list,
            particle_2=ref_list,
            print_every=f'{print_every}',
            name=f'{name}'
        )  
          
    
    def finsh_uncomplete_equlibration_run(self, com_list, ref_list, stiff, xmin, xmax, n_windows, input_parameters, continue_run=False):
        self.equlibration_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        self.com_distance_observable(com_list, ref_list)
        no_traj_file = []
        for sim in self.equlibration_sims:
            if not hasattr(sim.sim_files, 'traj'):
                no_traj_file.append(sim)
        if not no_traj_file:
            print('All windows have trajectory files')
            return None
        manager = SimulationManager()
        for sim, f_1, f_2 in zip(no_traj_file, self.umbrella_forces_1, self.umbrella_forces_2):
            if continue_run == False:
                sim.build(clean_build='force')
                sim.add_force(self.umbrella_force_1)
                sim.add_force(f_1)
                sim.add_force(f_2)
                #sim.add_observable(self.com_observable)
            sim.input_file(input_parameters)
            manager.queue_sim(sim, continue_run=continue_run)
        manager.worker_manager()
            
    def finsh_uncomplete_prouction_run(self, com_list, ref_list, stiff, xmin, xmax, n_windows, input_parameters, continue_run=False):
        self.equlibration_windows(n_windows)
        self.production_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        self.com_distance_observable(com_list, ref_list)
        no_traj_file = []
        for sim in self.production_sims:
            if not hasattr(sim.sim_files, 'traj'):
                no_traj_file.append(sim)
        if not no_traj_file:
            print('All windows have trajectory files')
            return None
        manager = SimulationManager()
        for sim, f_1, f_2 in zip(no_traj_file, self.umbrella_forces_1, self.umbrella_forces_2):
            if continue_run == False:
                sim.build(clean_build='force')
                sim.add_force(f_1)
                sim.add_force(f_2)
                sim.add_observable(self.com_observable)
            sim.input_file(input_parameters)
            manager.queue_sim(sim, continue_run=continue_run)
        manager.worker_manager()
    
    
#     def equlibration_windows(self, n_windows):
#         self.equlibration_sim_dir = join(self.system_dir, 'equlibration')
#         self.equlibration_file_dirs = join(self.equlibration_sim_dir, 'file_dirs')
#         if not exists(self.equlibration_sim_dir):
#             os.mkdir(self.equlibration_sim_dir)
#         if not exists(self.equlibration_file_dirs):
#             os.mkdir(self.equlibration_file_dirs)
#         else:
#             shutil.rmtree(self.equlibration_file_dirs)
            
#         n_conf = sep_conf(self.pull_sim.sim_files.last_conf,
#                           self.pull_sim.sim_files.traj,
#                           join(self.equlibration_sim_dir, 'file_dirs'))
        
#         self.equlibration_sims = []
#         for window in range(n_windows):
#             shutil.copyfile(self.pull_sim.sim_files.top,
#                             join(self.equlibration_sim_dir,'file_dirs', str(window), self.pull_sim.sim_files.top.split('/')[-1]))
#             self.equlibration_sims.append(Simulation(join(self.equlibration_file_dirs, str(window)), join(self.equlibration_sim_dir, str(window))))      
#     def eq_process(self, steps, input_parameters=None):
#         """
#         Run equlibration simulation on the inital conformation to ensure full relaxation.
        
#         Parameters:
#             steps (str): number of oxDNA time steps.
#             input_parameter (dict): optional modification of simulation input_parameters
#         """
#         self.eq_sim_dir = join(self.system_dir, 'eq')
#         self.eq_sim = Simulation(self.file_dir, self.eq_sim_dir)
#         self.eq_sim.build(clean_build='force')
#         self.eq_sim.input_file({'steps':str(steps)})
#         if input_parameters:    
#             self.eq_sim.input_file(input_parameters)
#         self.eq_sim.oxpy_run.run(join=True)
#         self.eq_sim.sim_files.parse_current_files()
#     def xmin_process(self, force_1, force_2, steps, input_parameters=None):
#         """
#         """
#         self.xmin_sim_dir = join(self.system_dir, 'xmin')
#         self.xmin_sim = Simulation(self.eq_sim_dir, self.xmin_sim_dir)
#         self.xmin_sim.build(clean_build='force')
#         self.xmin_sim.input_file({'steps':str(steps)})
#         self.xmin_sim.add_force(force_1)
#         self.xmin_sim.add_force(force_2)
#         if input_parameters:    
#             self.xmin_sim.input_file(input_parameters)
#         self.xmin_sim.oxpy_run.run(join=True)
#         self.xmin_sim.sim_files.parse_current_files()

#     def pull_process(self, force_1, force_2, n_windows, xmin, xmax, steps_per_conf, input_parameters=None):
#         steps = int(n_windows) * int(steps_per_conf)
#         force_rate = np.round((float(xmax) - float(xmin)) / steps, 12)
#         self.pull_sim_dir = join(self.system_dir, 'pull')
#         self.pull_sim = Simulation(self.xmin_sim_dir, self.pull_sim_dir)
#         self.pull_sim.build(clean_build='force')
#         self.pull_sim.input_file({'steps':f'{steps}',
#                                   'print_conf_interval': f'{int(steps_per_conf)}',
#                                   'print_energy_every': f'{int(steps_per_conf)}'})
#         force_1['force']['rate'] = str(force_rate)
#         force_2['force']['rate'] = str(force_rate)
#         self.pull_sim.add_force(force_1)
#         self.pull_sim.add_force(force_2)
#         if input_parameters:    
#             self.pull_sim.input_file(input_parameters)
#         self.pull_sim.oxpy_run.run(join=True)
#         self.pull_sim.sim_files.parse_current_files()
        
#     def prepare_window_confs(self, eq_steps, xmax_steps, steps_per_conf):
#         self.eq_process(eq_steps)
#         self.xmin_process(self.xmin_com_force_1,
#                           self.xmin_com_force_2,
#                           xmax_steps)
#         self.pull_process(self.pull_com_force_1,
#                           self.pull_com_force_2,
#                           self.n_windows,
#                           self.xmin,
#                           self.xmax,
#                           steps_per_conf)
        
    
    
    
    
    
    
    
#     def read_progress(self):
#         if exists(join(self.system_dir, 'eq')):
#             self.eq_sim_dir = join(self.system_dir, 'eq')
#             self.eq_sim = Simulation(self.file_dir, self.eq_sim_dir)
        
#         if exists(join(self.system_dir, 'xmin')):
#             self.xmin_sim_dir = join(self.system_dir, 'xmin')
#             self.xmin_sim = Simulation(self.file_dir, self.xmin_sim_dir)     
        
#         if exists(join(self.system_dir, 'pull')):
#             self.pull_sim_dir = join(self.system_dir, 'pull')
#             self.pull_sim = Simulation(self.file_dir, self.pull_sim_dir)    
        
#         if exists(join(self.system_dir, 'equlibration')):
#             self.equlibration_sim_dir = join(self.system_dir, 'equlibration')
#             self.equlibration_file_dirs = join(self.equlibration_sim_dir, 'file_dirs')
#             n_windows = len(os.listdir(self.equlibration_file_dirs))
#             self.equlibration_sims = []
#             for window in range(n_windows):
#                 self.equlibration_sims.append(Simulation(join(self.equlibration_file_dirs, str(window)), join(self.equlibration_sim_dir, str(window))))
                
#         if exists(join(self.system_dir, 'production')):
#             self.production_sim_dir = join(self.system_dir, 'production')
#             self.production_window_dirs = [join(self.production_sim_dir, str(window)) for window in range(n_windows)]
#             n_windows = len(self.equlibration_sims)
#             self.production_sims = []
#             for s, window_dir, window in zip(self.equlibration_sims, self.production_window_dirs, range(n_windows)):
#                 self.production_sims.append(Simulation(self.equlibration_sims[window].sim_dir, str(window_dir)))       

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

class ComUmbrellaSampling(BaseUmbrellaSampling):
    def __init__(self, file_dir, system, n_windows, xmin, xmax, com_list, ref_list):
        super().__init__(file_dir, system)
        self.f = Force()
        self.obs = Observable()
        self.file_dir = file_dir
        self.n_windows = n_windows
        self.xmin = xmin
        self.xmax = xmax
        self.com_list = com_list
        self.ref_list = ref_list

    def run_full_umbrella(self, eq_steps, xmax_steps, steps_per_conf, equlibration_steps, production_steps):
        self.com_observable(print_every=1e4, name='com_distance.txt')
        self.eq_process(eq_steps)
        self.xmin_process(self.xmin_com_force_1, self.xmin_com_force_2, xmax_steps)
        self.pull_process(self.pull_com_force_1, self.pull_com_force_2, self.n_windows, self.xmin, self.xmax, steps_per_conf)
        self.equlibration_windows(self.n_windows)
        
        manager = SimulationManager()
        for sim, umbrella_force_1, umbrella_force_2 in zip(self.equlibration_sims, self.umbrella_forces_1, self.umbrella_forces_2):
            sim.build(clean_build='force')
            sim.add_force(umbrella_force_1)
            sim.add_force(umbrella_force_2)
            sim.add_observable(self.com_distance_observable)
            sim.input_file({'steps':f'{equlibration_steps}', 'print_conf_interval':f'{equlibration_steps}', 'print_energy_every':f'{equlibration_steps}'})
            manager.queue_sim(sim)
        manager.worker_manager()
        for sim in self.equlibration_sims: sim.sim_files.parse_current_files()
        
        self.production_windows(self.n_windows)
        for sim, umbrella_force_1, umbrella_force_2 in zip(self.production_sims, self.umbrella_forces_1, self.umbrella_forces_2):
            sim.build(clean_build='force')
            sim.add_force(umbrella_force_1)
            sim.add_force(umbrella_force_2)
            sim.add_observable(self.com_distance_observable)
            sim.input_file({'steps':f'{production_steps}', 'print_conf_interval':f'{production_steps}', 'print_energy_every':f'{production_steps}'})
            manager.queue_sim(sim)
        manager.worker_manager()
        for sim in self.production_sims: sim.sim_files.parse_current_files()
    
    def gen_forces(self, xmin_stiff, pull_stiff, umbrella_stiff):
        self.xmin_force(xmin_stiff)
        self.pulling_force(pull_stiff)
        self.umbrella_forces(umbrella_stiff)
    
    def com_observable(self, print_every=1e4, name='com_distance.txt'):
        self.com_distance_observable = self.obs.distance(
            particle_1=self.com_list,
            particle_2=self.ref_list,
            print_every=f'{print_every}',
            name=f'{name}'
        )        
    
    def xmin_force(self, stiff):
        self.xmin_com_force_1 = self.f.com_force(
            com_list=self.com_list,                        
            ref_list=self.ref_list,                        
            stiff=stiff,                    
            r0=self.xmin,                       
            PBC='1',                         
            rate='0',
        )       
        self.xmin_com_force_2= self.f.com_force(
            com_list=self.ref_list,                        
            ref_list=self.com_list,                        
            stiff=self.xmin,                    
            r0=self.xmin,                       
            PBC='1',                         
            rate='0',
        )
    
    def pulling_force(self, stiff):
        self.pull_com_force_1 = self.f.com_force(
            com_list=self.com_list,                        
            ref_list=self.ref_list,                        
            stiff=stiff,                    
            r0=self.xmin,                       
            PBC='1',                         
            rate='0',
        )        
        self.pull_com_force_2 = self.f.com_force(
            com_list=self.ref_list,                        
            ref_list=self.com_list,                        
            stiff=stiff,                    
            r0=self.xmin,                       
            PBC='1',                         
            rate='0',
        )
    
    def umbrella_forces(self, stiff):
        x_range = np.linspace(self.xmin, self.xmax, (self.n_windows + 1))[1:]

        self.umbrella_forces_1 = []
        self.umbrella_forces_2 = []
        
        for x_val in x_range:   
            self.umbrella_force_1 = self.f.com_force(
                com_list=self.com_list,                        
                ref_list=self.ref_list,                        
                stiff=f'{stiff}',                    
                r0=f'{x_val}',                       
                PBC='1',                         
                rate='0',
            )        
            self.umbrella_forces_1.append(self.umbrella_force_1)
            
            self.umbrella_force_2= self.f.com_force(
                com_list=self.ref_list,                        
                ref_list=self.com_list,                        
                stiff=f'{stiff}',                    
                r0=f'{x_val}',                       
                PBC='1',                          
                rate='0',
            )
            self.umbrella_forces_2.append(self.umbrella_force_2)

            
    
def sep_conf(last_conf, traj, sim_dir):
    """
    Separate the trajectory file into separate files for each umbrella window.
    :param last_conf: path to last configuration file of pulling simulations
    :param traj: path to trajectory file of pulling simulations
    :param sim_dir: directory to save the separated files
    :return: None
    """
    # Create base umbrella simulation directory if it does not exist
    if not exists(sim_dir):
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
            c_dir = join(sim_dir, str(i))
            if not exists(c_dir):
                os.mkdir(c_dir)
            # Write the configuration to the new window
            with open(join(c_dir, f'conf_{i}.dat'), 'w') as f:
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
                with open(join(sim_dir, str(window), 'run.sh'), 'w') as r:
                    win = window + 1
                    for line in lines:
                        if 'job-name' in line:
                            r.write(f'#SBATCH --job-name="{window}"\n')
                        else:
                            r.write(line)
                    if file < (gpu_set - 1):
                        r.write(f'cd {join(sim_dir, str(win))}\nsbatch run.sh')
                window += 1
    return None


def run_production_slurm_files(n_gpus, n_windows, sim_dir):
    windows_per_gpu = get_windows_per_gpu(n_gpus, n_windows)
    run_dirs = [0]
    for gpu_set in windows_per_gpu:
        run_dirs.append(run_dirs[-1] + gpu_set)
    run_dirs.pop()
    for i in run_dirs:
        os.chdir(join(f"{sim_dir}", str(i)))
        os.system("sbatch run.sh")
        
