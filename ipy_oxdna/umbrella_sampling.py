from ipy_oxdna.oxdna_simulation import Simulation, Force, Observable, SimulationManager
from ipy_oxdna.wham_analysis import *
import multiprocessing as mp
import os
from os.path import join, exists
import numpy as np
import shutil
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal, norm
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import scienceplots
from copy import deepcopy
from ipy_oxdna.vmmc import VirtualMoveMonteCarlo
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed# from numba import jit
import pickle
from json import load, dump
import traceback


class BaseUmbrellaSampling:
    def __init__(self, file_dir, system, clean_build=False):
        self.clean_build = clean_build
        self.system = system
        self.file_dir = file_dir
        self.system_dir = join(self.file_dir, self.system)
        if not exists(self.system_dir):
            os.mkdir(self.system_dir)  
        
        self.windows = UmbrellaWindow(self)
        self.us_build = UmbrellaBuild(self)
        self.analysis = UmbrellaAnalysis(self)
        self.wham = WhamAnalysis(self)
        
        self.f = Force()
        self.obs = Observable()
        
        self.read_progress()
        
        self.umbrella_bias = None
        self.com_by_window = None
        self.r0 = None
    
    def queue_sims(self, simulation_manager, sim_list, continue_run=False):
        for sim in sim_list:
            simulation_manager.queue_sim(sim, continue_run=continue_run)        
     
     
    def spawn(self, f, args=(), join=False):
        """Spawn subprocess"""
        p = mp.Process(target=f, args=args)
        p.start()
        if join == True:
            p.join()
        self.process = p
        
    def spawn_wham_run(self, wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot, join=False):
        self.spawn(self.wham_run, args=(wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot), join=join)
               
    def wham_run(self, wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot, all_observables=False):
        """
        Run the weighted histogram analysis technique (Grossfield, Alan http://membrane.urmc.rochester.edu/?page_id=126)
        
        Parameters:
            wham_dir (str): directory in which wham is complied using the install_wham.sh script.
            
            xmin (str): smallest value of the order parameter to be sampled.
            
            xmax (str): largest value of the order parameter to be sample.
            
            umbrella_stiff (str): stiffness of the umbrella potential.
            
            n_bins (str): number of bins of the resultant free energy profile.
            
            tol (str): the tolerance of convergence for the WHAM technique.
            
            n_boot (str): number of monte carlo bootstrapping steps to take.
        """
        self.wham.wham_dir = wham_dir
        self.wham.xmin = xmin
        self.wham.xmax = xmax
        self.wham.umbrella_stiff = umbrella_stiff
        self.wham.n_bins = n_bins
        self.wham.tol = tol
        self.wham.n_boot = n_boot
        
        if all_observables is True:
            write_com_files(self)
        self.wham.run_wham(wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot)
        self.free = self.wham.to_si(n_bins, self.com_dir)
        self.mean = self.wham.w_mean(self.free)
        try:
            self.standard_error, self.confidence_interval = self.wham.bootstrap_w_mean_error(self.free)
        except:
            self.standard_error = 'failed'
            
    def convergence_analysis(self, n_chunks, data_added_per_iteration, wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot):
        """
        Split your data into a set number of chunks to check convergence.
        If all chunks are the same your free energy profile if probabily converged
        Also create datasets with iteraterativly more data, to check convergence progress
        """
        self.wham.wham_dir = wham_dir
        self.wham.xmin = xmin
        self.wham.xmax = xmax
        self.wham.umbrella_stiff = umbrella_stiff
        self.wham.n_bins = n_bins
        self.wham.tol = tol
        self.wham.n_boot = n_boot
        
        if not exists(self.com_dir):
            self.wham_run(wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot)
            
        self.convergence_dir = join(self.com_dir, 'convergence_dir')
        self.chunk_convergence_analysis_dir = join(self.convergence_dir, 'chunk_convergence_analysis_dir')
        self.data_truncated_convergence_analysis_dir = join(self.convergence_dir, 'data_truncated_convergence_analysis_dir')
        
        if exists(self.convergence_dir):
            shutil.rmtree(self.convergence_dir)
            
        if not exists(self.convergence_dir):
            os.mkdir(self.convergence_dir)
            os.mkdir(self.chunk_convergence_analysis_dir)
            os.mkdir(self.data_truncated_convergence_analysis_dir) 
  
        self.wham.chunk_convergence_analysis(n_chunks)
        
        self.chunk_dirs_free = [self.wham.to_si(self.wham.n_bins, chunk_dir) for chunk_dir in self.wham.chunk_dirs]
        self.chunk_dirs_mean = [self.wham.w_mean(free_energy) for free_energy in self.chunk_dirs_free]
        try:
            self.chunk_dirs_standard_error, self.chunk_dirs_confidence_interval = zip(
                *[self.wham.bootstrap_w_mean_error(free_energy) for free_energy in self.chunk_dirs_free]
            )
        except:
            self.chunk_dirs_standard_error = ['failed' for _ in range(len(self.chunk_dirs_free))]
            self.chunk_dirs_confidence_interval = ['failed' for _ in range(len(self.chunk_dirs_free))]


        self.wham.data_truncated_convergence_analysis(data_added_per_iteration)
        
        self.data_truncated_free = [self.wham.to_si(self.wham.n_bins, chunk_dir) for chunk_dir in self.wham.data_truncated_dirs]
        self.data_truncated_mean = [self.wham.w_mean(free_energy) for free_energy in self.data_truncated_free]
        try:
            self.data_truncated_standard_error, self.data_truncated_confidence_interval = zip(
                *[self.wham.bootstrap_w_mean_error(free_energy) for free_energy in self.data_truncated_free]
            )
        except:
            self.data_truncated_standard_error = ['failed' for _ in range(len(self.data_truncated_free))]
            self.data_truncated_confidence_interval = ['failed' for _ in range(len(self.data_truncated_free))]
        
        return None
                                    
    
    def read_progress(self):
        self.read_pre_equlibration_progress()
        self.read_equlibration_progress()
        self.read_production_progress()
        self.read_wham_progress()
        self.read_convergence_analysis_progress()
        self.read_melting_temperature_progress()
        
    def read_pre_equlibration_progress(self):
        if exists(join(self.system_dir, 'pre_equlibration')):
            self.pre_equlibration_sim_dir = join(self.system_dir, 'pre_equlibration')
            n_windows = len(os.listdir(self.pre_equlibration_sim_dir))
            self.pre_equlibration_sims = []
            for window in range(n_windows):
                self.pre_equlibration_sims.append(Simulation(join(self.pre_equlibration_sim_dir, str(window)), join(self.pre_equlibration_sim_dir, str(window))))

    def read_equlibration_progress(self):
        if exists(join(self.system_dir, 'equlibration')):
            self.equlibration_sim_dir = join(self.system_dir, 'equlibration')
            self.n_windows = len(os.listdir(self.equlibration_sim_dir))
            self.equlibration_sims = []
            for window in range(self.n_windows):
                self.equlibration_sims.append(Simulation(join(self.equlibration_sim_dir, str(window)), join(self.equlibration_sim_dir, str(window))))
    
    def read_production_progress(self):              
        if exists(join(self.system_dir, 'production')):
            self.production_sim_dir = join(self.system_dir, 'production')
            n_windows = len(self.equlibration_sims)
            self.production_window_dirs = [join(self.production_sim_dir, str(window)) for window in range(n_windows)]
            
            self.production_sims = []
            for s, window_dir, window in zip(self.equlibration_sims, self.production_window_dirs, range(n_windows)):
                self.production_sims.append(Simulation(self.equlibration_sims[window].sim_dir, str(window_dir))) 
    
    def read_wham_progress(self):          
        if exists(join(self.system_dir, 'production', 'com_dir', 'freefile')):
            self.com_dir = join(self.system_dir, 'production', 'com_dir')
            with open(join(self.production_sim_dir, 'com_dir', 'freefile'), 'r') as f:
                file = f.readlines()
            file = [line for line in file if not line.startswith('#')]
            self.n_bins = len(file)
            
            with open(join(self.com_dir, 'metadata'), 'r') as f:
                lines = [line.split(' ') for line in f.readlines()]
            
            self.wham.xmin = float(lines[0][1])
            self.wham.xmax = float(lines[-1][1])
            self.wham.umbrella_stiff = float(lines[0][-1])
            self.wham.n_bins = self.n_bins
            # self.wham.get_n_data_per_com_file()
            self.free = self.wham.to_si(self.n_bins, self.com_dir)
            self.mean = self.wham.w_mean(self.free)
            try:
                self.standard_error, self.confidence_interval = self.wham.bootstrap_w_mean_error(self.free)
            except:
                self.standard_error, self.confidence_interval = ('failed', 'failed')
        
    def read_convergence_analysis_progress(self):   
        if exists(join(self.system_dir, 'production', 'com_dir', 'convergence_dir')):
            try:
                self.convergence_dir = join(self.com_dir, 'convergence_dir')
                self.chunk_convergence_analysis_dir = join(self.convergence_dir, 'chunk_convergence_analysis_dir')
                self.data_truncated_convergence_analysis_dir = join(self.convergence_dir, 'data_truncated_convergence_analysis_dir')  
                self.wham.chunk_dirs = [join(self.chunk_convergence_analysis_dir, chunk_dir) for chunk_dir in os.listdir(self.chunk_convergence_analysis_dir)]
                self.wham.data_truncated_dirs = [join(self.data_truncated_convergence_analysis_dir, chunk_dir) for chunk_dir in os.listdir(self.data_truncated_convergence_analysis_dir)]
                
                self.chunk_dirs_free = [self.wham.to_si(self.n_bins, chunk_dir) for chunk_dir in self.wham.chunk_dirs]
                self.chunk_dirs_mean = [self.wham.w_mean(free_energy) for free_energy in self.chunk_dirs_free]
                try:
                    self.chunk_dirs_standard_error, self.chunk_dirs_confidence_interval = zip(
                        *[self.wham.bootstrap_w_mean_error(free_energy) for free_energy in self.chunk_dirs_free]
                    )
                except:
                    self.chunk_dirs_standard_error = ['failed' for _ in range(len(self.chunk_dirs_free))]
                    self.chunk_dirs_confidence_interval = ['failed' for _ in range(len(self.chunk_dirs_free))]
    
                    
                self.data_truncated_free = [self.wham.to_si(self.n_bins, chunk_dir) for chunk_dir in self.wham.data_truncated_dirs]
                self.data_truncated_mean = [self.wham.w_mean(free_energy) for free_energy in self.data_truncated_free]
                try:
                    self.data_truncated_standard_error, self.data_truncated_confidence_interval = zip(
                        *[self.wham.bootstrap_w_mean_error(free_energy) for free_energy in self.data_truncated_free]
                    )
                except:
                    self.data_truncated_standard_error = ['failed' for _ in range(len(self.data_truncated_free))]
                    self.data_truncated_confidence_interval = ['failed' for _ in range(len(self.data_truncated_free))]
            except:
                pass
    
    def read_melting_temperature_progress(self):
        if exists(join(self.system_dir, 'production', 'vmmc_dir')):
            self.vmmc_dir = join(self.system_dir, 'production', 'vmmc_dir')
            self.vmmc_sim = VirtualMoveMonteCarlo(self.file_dir, self.vmmc_dir)
            self.vmmc_sim.analysis.read_vmmc_op_data()
            self.vmmc_sim.analysis.calculate_sampling_and_probabilities()
            self.vmmc_sim.analysis.calculate_and_estimate_melting_profiles()
                
    def get_biases(self):
        #I need to get the freefile
        #The free file is in the com_dim
        with open(f'{self.com_dir}/freefile', 'r') as f:
            data = f.readlines()
            
        for idx, line in enumerate(data):
            if '#Window' in line:
                break
            else:
                pass
        data = data[idx:]
        data = [line[1:] for line in data]
        data = [line.replace('\t', ' ') for line in data]
        data = data[1:]
        data = [line.split() for line in data]
        data = [np.double(line[1]) for line in data]
 
        return data
   

   
    def get_com_distance_by_window(self):
        com_distance_by_window = {}
        for idx,sim in enumerate(self.production_sims):
            sim.sim_files.parse_current_files()
            df = pd.read_csv(sim.sim_files.com_distance, header=None, engine='pyarrow', dtype=np.double)
            com_distance_by_window[idx] = df
        self.com_by_window = com_distance_by_window

    def get_r0_values(self):
        self.r0 = []
        with open(join(self.com_dir, 'metadata'), 'r') as f:
            for line in f:
                self.r0.append(float(line.split(' ')[1]))
                
    def get_temperature(self):
        pre_temp = self.production_sims[0].input.input['T']
        if ('C'.upper() in pre_temp) or ('C'.lower() in pre_temp):
            self.temperature = (float(pre_temp[:-1]) + 273.15) / 3000
        elif ('K'.upper() in pre_temp) or ('K'.lower() in pre_temp):
             self.temperature = float(pre_temp[:-1]) / 3000
             
    def get_n_windows(self):
        
        try:
            self.n_windows = len(self.production_sims)
        except:
            pass
        try:
            self.n_windows = len(self.equlibration_sims)
        except:
            pass
        try:
            self.n_windows = len(self.pre_equlibration_sims)
        except:
            print('No simulations found')
    
    def get_bias_potential_value(self, xmin, xmax, n_windows, stiff):
        x_range = np.round(np.linspace(xmin, xmax, (n_windows + 1), dtype=np.double)[1:], 3)
        umbrella_bias = [0.5 * np.double(stiff) * (com_values - eq_pos)**2 for com_values, eq_pos in zip(self.com_by_window.values(), x_range)]
        self.umbrella_bias = umbrella_bias
    
    def copy_last_conf_from_eq_to_prod(self):
        for eq_sim, prod_sim in zip(self.equlibration_sims, self.production_sims):
            shutil.copyfile(eq_sim.sim_files.last_conf, f'{prod_sim.sim_dir}/last_conf.dat')
            

            
class UmbrellaBuild:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
    
    def build(self, sims, input_parameters, forces_list, observables_list,
              observable=False, sequence_dependant=False, cms_observable=False, protein=None, force_file=None):
        
        if exists(join(self.base_umbrella.system_dir, 'production')):
            if self.base_umbrella.clean_build is True:
                answer = input('Are you sure you want to delete all simulation files? Type y/yes to continue or anything else to return use UmbrellaSampling(clean_build=str(force) to skip this message')
                if answer.lower() not in ['y', 'yes']:
                    sys.exit('\nRemove optional argument clean_build and continue a previous umbrella simulation using:\nsimulation_manager.run(continue_run=int(n_steps))')
            elif self.base_umbrella.clean_build == False:
                sys.exit('\nThe simulation directory already exists, if you wish to write over the directory set:\nUmbrellaSampling(clean_build=str(force)).\n\nTo continue a previous umbrella simulation use:\nsimulation_manager.run(continue_run=int(n_steps))')

        # Using ThreadPoolExecutor to parallelize the simulation building
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._build_simulation, sim, forces, input_parameters, observables_list,
                                       observable, sequence_dependant, cms_observable, protein, force_file)
                       for sim, forces in zip(sims, forces_list)]

            for future in futures:
                try:
                    future.result()  # Wait for each future to complete and handle exceptions
                except Exception as e:
                    print(f"An error occurred: {e}")
        self.parallel_force_group_name(sims)
    
    
    def _build_simulation(self, sim, forces, input_parameters, observables_list,
                          observable, sequence_dependant, cms_observable, protein, force_file):
        try:
            sim.build(clean_build='force')
            sim.input_file(input_parameters)
            if protein is not None:
                sim.add_protein_par()
            if force_file is not None:
                sim.add_force_file()
            for force in forces:
                sim.add_force(force)
            if observable:
                for observables in observables_list:
                    sim.add_observable(observables)
            if cms_observable is not False:
                for cms_obs_dict in cms_observable:
                    sim.oxpy_run.cms_obs(cms_obs_dict['idx'],
                                         name=cms_obs_dict['name'],
                                         print_every=cms_obs_dict['print_every'])
            if sequence_dependant:
                sim.sequence_dependant()
            sim.sim_files.parse_current_files()
        except Exception as e:
            error_traceback = traceback.format_exc()  # Gets the full traceback
            print(f"Build error in simulation {sim.sim_dir}: {e}\nTraceback: {error_traceback}")
            raise

    def parallel_force_group_name(self, sims):
        with ThreadPoolExecutor() as executor:
            executor.map(self.process_simulation, sims)

    def process_simulation(self, sim):
        sim.sim_files.parse_current_files()
        with open(sim.sim_files.force, 'r') as f:
            force_js = load(f)
        force_js_modified = {key: {'group_name': key, **value} for key, value in force_js.items()}
        with open(sim.sim_files.force, 'w') as f:
            dump(force_js_modified, f, indent=4)


class ComUmbrellaSampling(BaseUmbrellaSampling):
    def __init__(self, file_dir, system, clean_build=False):
        super().__init__(file_dir, system, clean_build=clean_build)
        self.observables_list = []
     
    def build_pre_equlibration_runs(self, simulation_manager,  n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters, starting_r0, steps, observable=False, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False, protein=None, force_file=None):
        self.observables_list = []
        self.windows.pre_equlibration_windows(n_windows)
        self.rate_umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows, starting_r0, steps)
        self.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        if continue_run is False:
            self.us_build.build(self.pre_equlibration_sims, input_parameters,
                                self.forces_list, self.observables_list,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file)
        self.queue_sims(simulation_manager, self.pre_equlibration_sims, continue_run=continue_run)
    
    def build_equlibration_runs(self, simulation_manager,  n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                                observable=False, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False,
                                protein=None, force_file=None):
        """
        Build the equlibration run
        
        Parameters:
            simulation_manager (SimulationManager): pass a simulation manager.
            
            n_windows (int): number of simulation windows.
            
            com_list (str): comma speperated list of nucleotide indexes.
            
            ref_list (str): comma speperated list of nucleotide indexex.
            
            stiff (float): stiffness of the umbrella potential.
            
            xmin (float): smallest value of the order parameter to be sampled.
            
            xmax (float): largest value of the order parameter to be sampled.
            
            input_parameters (dict): dictonary of oxDNA parameters.
            
            observable (bool): Boolean to determine if you want to print the observable for the equlibration simulations.
            
            sequence_dependant (bool): Boolean to use sequence dependant parameters.
            
            print_every (float): how often to print the order parameter.
            
            name (str): depreciated.
            
            continue_run (float): number of steps to continue umbrella sampling (i.e 1e7).
            
            protein (bool): Use a protein par file and run with the ANM oxDNA interaction potentials.
            
            force_file (bool): Add an external force to the umbrella simulations.
        """
        self.observables_list = []
        self.n_windows = n_windows
        self.windows.equlibration_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        self.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        if continue_run is False:
            self.us_build.build(self.equlibration_sims, input_parameters,
                                self.forces_list, self.observables_list,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file)
        self.queue_sims(simulation_manager, self.equlibration_sims, continue_run=continue_run)
        
        
    def build_production_runs(self, simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                              observable=True, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False,
                              protein=None, force_file=None):
        """
        Build the production run
        
        Parameters:
            simulation_manager (SimulationManager): pass a simulation manager.
            
            n_windows (int): number of simulation windows.
            
            com_list (str): comma speperated list of nucleotide indexes.
            
            ref_list (str): comma speperated list of nucleotide indexex.
            
            stiff (float): stiffness of the umbrella potential.
            
            xmin (float): smallest value of the order parameter to be sampled.
            
            xmax (float): largest value of the order parameter to be sampled.
            
            input_parameters (dict): dictonary of oxDNA parameters.
            
            observable (bool): Boolean to determine if you want to print the observable for the equlibration simulations, make sure to not have this as false for production run.
            
            sequence_dependant (bool): Boolean to use sequence dependant parameters.
            
            print_every (float): how often to print the order parameter.
            
            name (str): depreciated.
            
            continue_run (float): number of steps to continue umbrella sampling (i.e 1e7).
            
            protein (bool): Use a protein par file and run with the ANM oxDNA interaction potentials.
            
            force_file (bool): Add an external force to the umbrella simulations.
        """
        self.observables_list = []
        self.windows.equlibration_windows(n_windows)
        self.windows.production_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        self.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        if continue_run is False:
            self.us_build.build(self.production_sims, input_parameters,
                                self.forces_list, self.observables_list,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file) 
        
        self.queue_sims(simulation_manager, self.production_sims, continue_run=continue_run)         
    
    def com_distance_observable(self, com_list, ref_list,  print_every=1e4, name='com_distance.txt'):
        """ Build center of mass observable"""
        obs = Observable()
        com_observable = obs.distance(
            particle_1=com_list,
            particle_2=ref_list,
            print_every=f'{print_every}',
            name=f'{name}',
            PBC='1'
        )  
        self.observables_list.append(com_observable)
 

    def umbrella_forces(self, com_list, ref_list, stiff, xmin, xmax, n_windows):
        """ Build Umbrella potentials"""
        x_range = np.round(np.linspace(xmin, xmax, (n_windows + 1))[1:], 3)
        umbrella_forces_1 = []
        umbrella_forces_2 = []
        
        for x_val in x_range:   
            self.umbrella_force_1 = self.f.com_force(
                com_list=com_list,                        
                ref_list=ref_list,                        
                stiff=f'{stiff}',                    
                r0=f'{x_val}',                       
                PBC='1',                         
                rate='0',
            )        
            umbrella_forces_1.append(self.umbrella_force_1)
            
            self.umbrella_force_2= self.f.com_force(
                com_list=ref_list,                        
                ref_list=com_list,                        
                stiff=f'{stiff}',                    
                r0=f'{x_val}',                       
                PBC='1',                          
                rate='0',
            )
            umbrella_forces_2.append(self.umbrella_force_2)  
        self.forces_list = np.transpose(np.array([umbrella_forces_1, umbrella_forces_2]))     

    def rate_umbrella_forces(self, com_list, ref_list, stiff, xmin, xmax, n_windows, starting_r0, steps):
        """ Build Umbrella potentials"""
        
        x_range = np.round(np.linspace(xmin, xmax, (n_windows + 1))[1:], 3)
        
        umbrella_forces_1 = []
        umbrella_forces_2 = []
        
        for x_val in x_range:   
            force_rate_for_x_val = (x_val - starting_r0) / steps
            
            self.umbrella_force_1 = self.f.com_force(
                com_list=com_list,                        
                ref_list=ref_list,                        
                stiff=f'{stiff}',                    
                r0=f'{starting_r0}',                       
                PBC='1',                         
                rate=f'{force_rate_for_x_val}',
            )        
            umbrella_forces_1.append(self.umbrella_force_1)
            
            self.umbrella_force_2= self.f.com_force(
                com_list=ref_list,                        
                ref_list=com_list,                        
                stiff=f'{stiff}',                    
                r0=f'{starting_r0}',                       
                PBC='1',                          
                rate=f'{force_rate_for_x_val}',
            )
            umbrella_forces_2.append(self.umbrella_force_2)  
        self.forces_list = np.transpose(np.array([umbrella_forces_1, umbrella_forces_2])) 
        

        
    
    def fig_ax(self):
        self.ax = self.wham.plt_fig(title='Free Energy Profile', xlabel='End-to-End Distance (nm)', ylabel='Free Energy / k$_B$T')
    
    
    def plot_free(self, ax=None, title='Free Energy Profile', c=None, label=None, fmt=None):
        self.wham.plot_free_energy(ax=ax, title=title, label=label)
    
    def plot_free_mod(self, negative=False, ax=None, title='Free Energy Profile', c=None, label=None):
        self.wham.plot_free_energy_mod(negative=negative ,ax=ax, title=title, label=label)


        
class CustomObservableUmbrellaSampling(ComUmbrellaSampling):
    def __init__(self, file_dir, system, clean_build=False):
        super().__init__(file_dir, system, clean_build=clean_build)
        self.wham = CustomObsWham(self)
    
    def build_equlibration_runs(self, simulation_manager,  n_windows, 
                                com_list, ref_list, stiff, xmin, xmax,
                                input_parameters, cms_observable,
                                observable=False, sequence_dependant=False,
                                print_every=1e4, name='com_distance.txt', continue_run=False,
                                protein=None, force_file=None):
        self.observables_list = []
        self.windows.equlibration_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        self.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        if continue_run is False:
            self.us_build.build(self.equlibration_sims, input_parameters,
                                self.forces_list, self.observables_list,
                                observable=observable, sequence_dependant=sequence_dependant,
                                cms_observable=cms_observable, protein=protein, force_file=force_file)
        self.queue_sims(simulation_manager, self.equlibration_sims, continue_run=continue_run)
          
    def build_production_runs(self, simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax,
                              input_parameters,cms_observable, observable=True,
                              sequence_dependant=False, print_every=1e4,
                              name='com_distance.txt', continue_run=False,
                              protein=None, force_file=None):
        self.observables_list = []
        self.windows.equlibration_windows(n_windows)
        self.windows.production_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        self.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        if continue_run is False:
            self.us_build.build(self.production_sims, input_parameters,
                                self.forces_list, self.observables_list,
                                observable=observable, sequence_dependant=sequence_dependant,
                                cms_observable=cms_observable, protein=protein, force_file=force_file)
        self.queue_sims(simulation_manager, self.production_sims, continue_run=continue_run)   
    
    

class MeltingUmbrellaSampling(ComUmbrellaSampling):
    def __init__(self, file_dir, system, clean_build=False):
        super().__init__(file_dir, system, clean_build=clean_build)
        self.hb_by_window = None
        self.potential_energy_by_window = None
    
    def build_pre_equlibration_runs(self, simulation_manager,  n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters, starting_r0, steps, observable=False, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False, protein=None, force_file=None, custom_observable=False):
        self.observables_list = []
        self.windows.pre_equlibration_windows(n_windows)
        self.rate_umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows, starting_r0, steps)
        
        if observable:
            self.initialize_observables(com_list, ref_list, print_every, name)

        
        if continue_run is False:
            self.us_build.build(self.pre_equlibration_sims, input_parameters,
                                self.forces_list, self.observables_list,cms_observable=custom_observable,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file)
            for sim in self.pre_equlibration_sims:
                sim.build_sim.build_hb_list_file(com_list, ref_list)
        self.queue_sims(simulation_manager, self.pre_equlibration_sims, continue_run=continue_run)
        
    def build_equlibration_runs(self, simulation_manager,  n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                                observable=False, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False,
                                protein=None, force_file=None, custom_observable=False):
        self.observables_list = []
        self.windows.equlibration_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)

        if observable:
            self.initialize_observables(com_list, ref_list, print_every, name)
        
        if continue_run is False:
            self.us_build.build(self.equlibration_sims, input_parameters,
                                self.forces_list, self.observables_list, cms_observable=custom_observable,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file)
            for sim in self.equlibration_sims:
                sim.build_sim.build_hb_list_file(com_list, ref_list)
        self.queue_sims(simulation_manager, self.equlibration_sims, continue_run=continue_run)
        
        
    def build_production_runs(self, simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                              observable=True, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False,
                              protein=None, force_file=None, custom_observable=False):
        self.observables_list = []
        self.windows.equlibration_windows(n_windows)
        self.windows.production_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)

        if observable:
            self.initialize_observables(com_list, ref_list, print_every, name)

        if continue_run is False:
            self.us_build.build(self.production_sims, input_parameters,
                                self.forces_list, self.observables_list, cms_observable=custom_observable,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file)
            for sim in self.production_sims:
                sim.build_sim.build_hb_list_file(com_list, ref_list)
        self.queue_sims(simulation_manager, self.production_sims, continue_run=continue_run)   
        
    def initialize_observables(self, com_list, ref_list, print_every=1e4, name='all_observables.txt'):
        self.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        self.hb_list_observable(print_every=print_every, only_count='true', name=name)
        
        number_of_forces = self.get_n_external_forces()      
        for idx in range(number_of_forces):
            self.force_energy_observable(print_every=print_every, name=name, print_group=f'force_{idx}')
                    
        self.kinetic_energy_observable(print_every=print_every, name=name)
        self.potential_energy_observable(print_every=print_every, name=name, split='True')
    
    def hb_list_observable(self, print_every=1e4, name='hb_observable.txt', only_count='true'):
        """ Build center of mass observable"""
        hb_obs = self.obs.hb_list(
            print_every=str(print_every),
            name=name,
            only_count='true'
           )
        self.observables_list.append(hb_obs)
        
    def force_energy_observable(self, print_every=1e4, name='force_energy.txt', print_group=None):
        """_summary_

        Args:
            print_every (_type_, optional): _description_. Defaults to 1e4.
            name (str, optional): _description_. Defaults to 'force_energy.txt'.
        """
        force_energy_obs = self.obs.force_energy(
            print_every=str(print_every),
            name=str(name),
            print_group=str(print_group)
        )
        self.observables_list.append(force_energy_obs)
    
    def kinetic_energy_observable(self, print_every=1e4, name='kinetic_energy.txt'):
        kin_obs = self.obs.kinetic_energy(
            print_every=str(print_every),
            name=str(name)
        )
        self.observables_list.append(kin_obs)
        
    def potential_energy_observable(self, print_every=1e4, name='potential_energy.txt', split='True'):
        """ Build potential energy observable for temperature interpolation"""
        pot_obs = self.obs.potential_energy(
            print_every=str(print_every),
            split=str(split),
            name=str(name)
        )
        self.observables_list.append(pot_obs)
        
    def get_n_external_forces(self):
        try:
            force_file = [file for file in os.listdir(self.file_dir) if file.endswith('force.txt')][0]
            force_file = f'{self.file_dir}/{force_file}'
        
            number_of_forces = 0

            with open(force_file, 'r') as f:
                for line in f:
                    if '{' in line:
                        number_of_forces += 1
        except:
            number_of_forces = 0
            
        return number_of_forces + 2
        
    def temperature_interpolation(self, xmin, xmax, max_hb, temp_range, reread_files=False, all_observables=False, convergence_slice=None, molcon=None):

        if reread_files is False:
            if self.obs_df is None:
                self.analysis.read_all_observables('prod')
        elif reread_files is True:
            self.analysis.read_all_observables('prod')

        number_of_forces = self.get_n_external_forces()
        force_energy = [f'force_energy_{idx}' for idx in range(number_of_forces)]
        # force_energy = ['force_energy']

        min_length = min([len(inner_list['com_distance']) for inner_list in self.obs_df])
        truncated_com_values = [inner_list['com_distance'][:min_length] for inner_list in self.obs_df] 
        x_range = np.round(np.linspace(xmin, xmax, (self.n_windows + 1), dtype=np.double)[1:], 3)
        names = ['backbone', 'bonded_excluded_volume', 'stacking', 'nonbonded_excluded_volume', 'hydrogen_bonding', 'cross_stacking', 'coaxial_stacking', 'debye_huckel']
        truncated_potential_energy = [inner_list[names][:min_length] for inner_list in self.obs_df]
        truncated_kinetic_energy = [inner_list['kinetic_energy'][:min_length] for inner_list in self.obs_df]
        truncated_force_energy = [inner_list[force_energy][:min_length] for inner_list in self.obs_df]
        truncated_hb_values = [inner_list['hb_list'][:min_length] for inner_list in self.obs_df]
        hb_by_window = np.array(truncated_hb_values)
        # truncated_umbrella_bias = [0.5 * np.double(self.wham.umbrella_stiff) * (com_values - eq_pos)**2 for com_values, eq_pos in zip(truncated_com_values, x_range)]


        if convergence_slice is not None:
            truncated_com_values = [inner_list[convergence_slice] for inner_list in truncated_com_values] 
            # truncated_umbrella_bias = [inner_list[convergence_slice] for inner_list in truncated_umbrella_bias]
            names = ['backbone', 'bonded_excluded_volume', 'stacking', 'nonbonded_excluded_volume', 'hydrogen_bonding', 'cross_stacking', 'coaxial_stacking', 'debye_huckel']
            truncated_potential_energy = [inner_list[names][convergence_slice] for inner_list in truncated_potential_energy]
            truncated_kinetic_energy = [inner_list[convergence_slice] for inner_list in truncated_kinetic_energy]
            truncated_force_energy = [inner_list[force_energy][convergence_slice] for inner_list in truncated_force_energy]
            truncated_hb_values = [inner_list[convergence_slice] for inner_list in truncated_hb_values]
            hb_by_window = np.array(truncated_hb_values)
        # w_i = self.weight_sample()
        
        hb_by_window = np.where(hb_by_window <= max_hb, hb_by_window, max_hb)
        index_to_add_at = hb_by_window
        
        temperature = np.array(self.temperature, dtype=np.double)
        temp_range_scaled = self.celcius_to_scaled(temp_range)
        beta_range = 1 / temp_range_scaled
        beta = 1 / temperature
        bias = [[[] for _ in range(len(temp_range))] for _ in range(len(self.obs_df))]

        top_file = self.production_sims[0].sim_files.top
        with open(top_file, 'r') as f:
            n_particles_in_system = np.double(f.readline().split(' ')[0])
        
        n_particles_in_op = np.double(max_hb * 2)
                
        # truncated_potential_energy = [n_particles_in_op * innerlist for innerlist in truncated_potential_energy]
        # truncated_kinetic_energy = np.array(truncated_kinetic_energy) * n_particles_in_op
        # truncated_force_energy = np.array(truncated_force_energy) * n_particles_in_op
         
        truncated_non_pot_energy = truncated_kinetic_energy + truncated_force_energy

        energy_bias_per_window_per_temperature = np.array(self._new_calcualte_bias_energy(truncated_non_pot_energy, temp_range, truncated_potential_energy=truncated_potential_energy))
        
        energy_bias_per_window_per_temperature = [n_particles_in_system * innerlist for innerlist in energy_bias_per_window_per_temperature]
        
        for win_idx, temperature_bias in enumerate(energy_bias_per_window_per_temperature):
            for temp_idx, temp_bias in enumerate(temperature_bias):
                bias[win_idx][temp_idx].append(temp_bias)
        
        bias = np.array(bias).squeeze(2)        
        self.bias = bias

        beta_u_hb_list = [[[[] for _ in range(max_hb + 1)] for _ in range(len(temp_range))] for _ in range(len(self.obs_df))] 

        for win_idx, (b_u_win, hb_win) in enumerate(zip(bias, index_to_add_at)):
            
            for temp_idx, b_u_temp in enumerate(b_u_win):

                for b_u, hb in zip(b_u_temp, hb_win):
                    
                    beta_u_hb_list[win_idx][temp_idx][int(hb)].append(b_u)
                    
        self.beta_u_hb_list = beta_u_hb_list
        
        log_e_beta_u = np.empty((len(self.obs_df), len(temp_range), max_hb+1), dtype=np.double)

        for win_idx, b_u_hb_win in enumerate(beta_u_hb_list):
            
            for temp_idx, b_u_hb_temp in enumerate(b_u_hb_win):

                for hb_idx, b_u_hb in enumerate(b_u_hb_temp):
                    
                    if len(b_u_hb) > 0:
                        log_e_beta_u[win_idx][temp_idx][hb_idx] = logsumexp(b_u_hb)
                    else:
                        log_e_beta_u[win_idx][temp_idx][hb_idx] = 0
        self.log_e_beta_u = log_e_beta_u
        
        
        # f_i = self.get_biases()
        f_i = self.F_i
        weight = -beta_range[:, np.newaxis] * np.array(f_i)
        weight_norm = logsumexp(weight)
        A_i = weight - weight_norm
        
        self.com_max = np.max(np.array([com_dist for com_dist in truncated_com_values]))
        self.production_sims[0].build_sim.get_last_conf_top()
        init_dat = self.production_sims[0].build_sim.dat
        init_dat_location = f'{self.production_sims[0].file_dir}/{init_dat}'
        with open(init_dat_location, 'r') as f:
            next(f)
            box_info = f.readline().split(' ')
            self.box_size = float(box_info[-1].strip())
        
        
        if molcon is not None:
            box_size = self.molar_concentration_to_box_size(molcon)
            self.volume_correction = list(map(self.volume_correction, box_size))
        else:    
            self.volume_correction = np.log((((self.box_size / 2) * np.sqrt(3))**3) / ((4/3) * np.pi * (self.box_size / 2)**3))
                
        log_p_i_h = self.log_e_beta_u - logsumexp(self.log_e_beta_u, axis=2, keepdims=True)
        self.log_p_i_h = log_p_i_h

        a_log_p_i_h = log_p_i_h.T + A_i

        combine_log_p_i_h = logsumexp(a_log_p_i_h, axis=2)
        self.maybe_prob_discrete = np.exp(combine_log_p_i_h)
        
        free_energy = -combine_log_p_i_h.T
        free_energy -= free_energy.min(axis=1, keepdims=True)
        
        if molcon is not None:
            free_energy[:,0] = free_energy[:,0] - self.volume_correction
            self.free_energy_discrete = free_energy + self.volume_correction
            
        free_energy[:,0] = free_energy[:,0] - self.volume_correction
        self.free_energy_discrete = free_energy + self.volume_correction
        
        normed_free_energy = -free_energy - logsumexp(-free_energy, axis=1, keepdims=True)
        self.prob_discrete = np.exp(normed_free_energy)
        
        return self.free_energy_discrete, self.prob_discrete
    
    def volume_correction(self, box_size):
        return np.log((((box_size / 2) * np.sqrt(3))**3) / ((4/3) * np.pi * (self.box_size / 2)**3))

    def molar_concentration_to_box_size(self, molcon):
        box = ((2/(molcon*6.0221415*10**23))**(1/3)) / 8.5179*10.0**(9)
        return box
    
    def _new_calcualte_bias_energy(self, umbrella_bias, temperature_range, truncated_potential_energy=None):
        #Constants
        STCK_FACT_EPS_OXDNA2 = 2.6717
        STCK_BASE_EPS_OXDNA2 = 1.3523
        
        # Initialize the variables
        old_temperature = self.temperature
        
        # temperature_range = np.array((range(30, 70, 2)))
        new_temperatures = (temperature_range + 273.15) / 3000
        
        if truncated_potential_energy is not None:
            e_state = [states.sum(axis=1) for states in truncated_potential_energy]
            e_stack = [states['stacking'] for states in truncated_potential_energy]
        else:    
            e_state = [states.sum(axis=1) for states in self.potential_energy_by_window.values()]
            e_stack = [states['stacking'] for states in self.potential_energy_by_window.values()]
        
        #e_ext =   # Replace with actual value (external energy term)
        am = 1  # Replace with actual value (the increment amount)
        w = 0  # Replace with actual value (the weight)
        # umbrella_bias = np.zeros_like(umbrella_bias)
        
        # Initialize a list to store the results
        results_list = []
        
        for window_idx, (e_state_window, e_stack_window, e_ext) in enumerate(zip(e_state, e_stack, umbrella_bias)):
            n_data_points = len(e_state_window)
            n_temps = len(new_temperatures)
        
            # Initialize a NumPy array for this window
            results_window = np.zeros((n_temps, n_data_points))
        
            # Calculate et for the old temperature (simtemp in C++)
            et_old = (e_stack_window * STCK_FACT_EPS_OXDNA2) / (STCK_BASE_EPS_OXDNA2 + old_temperature * STCK_FACT_EPS_OXDNA2)
        
            # Calculate e0 for the old temperature
            e0_old = e_state_window - old_temperature * et_old
            # print(f'{e_ext=}')
            for temp_idx, new_temp in enumerate(new_temperatures):
                # Calculate et for the new temperature
                et_new = (e_stack_window * STCK_FACT_EPS_OXDNA2) / (STCK_BASE_EPS_OXDNA2 + new_temp * STCK_FACT_EPS_OXDNA2)
        
                # Calculate e0 for the new temperature
                e0_new = e_state_window - new_temp * et_new
        
                # Calculate the expression inside exp() for the new temperature
                energy_term = -(e0_old + e_ext + new_temp * et_old) / new_temp + (e0_old + e_ext + old_temperature * et_old) / old_temperature
        
                # Update results_window
                results_window[temp_idx, :] = am * energy_term.values
        
            # Add results_window to results_list
            results_list.append(results_window)
        return results_list
        
    def sigmoid(self, x, L, x0, k, b):
        return L / (1 + np.exp(-k * (x - x0))) + b
    
    
    def calculate_melting_temperature(self, temp_range):
        probabilities = self.prob_discrete
        
        #probabilities: n_temps x n_hb
        
        bound_states = probabilities[:,1:].sum(axis=1)
        unbound_states = probabilities[:,0]
        ratio = bound_states / unbound_states 
    
        finf = 1. + 1. / (2. * ratio) - np.sqrt((1. + 1. / (2. * ratio))**2 - 1.)
        self.finf = finf
        
        self.inverted_finfs = 1 - finf
        
        # Fit the sigmoid function to the inverted data
        p0 = [max(self.inverted_finfs), np.median(temp_range), 1, min(self.inverted_finfs)]  # initial guesses for L, x0, k, b
        self.popt, _ = curve_fit(self.sigmoid, temp_range, self.inverted_finfs, p0, method='dogbox')
    
        # Generate fitted data
        self.x_fit = np.linspace(min(temp_range), max(temp_range), 500)
        self.y_fit = self.sigmoid(self.x_fit, *self.popt)
        
        
        idx = np.argmin(np.abs(self.y_fit - 0.5))
        self.Tm = self.x_fit[idx]
        
    # @jit(nopython=True)
    # def fast_histogram(self, all_com_values, temp_biases, bin_edges):
    #     result = []
    #     for temp_bias in temp_biases:
    #         temp_result = []
    #         for com_values, t_bias in zip(all_com_values, temp_bias):
    #             hist, _ = np.histogram(com_values, bins=bin_edges, weights=t_bias)
    #             temp_result.append(hist)
    #         result.append(temp_result)
    #     return result   
        
    # # @jit(nopython=True)
    # # def compute_f_i_temps(self, f_i_temps_old, window_biases, beta_range, summed_p_i_b_s, numerator, f_i_bias_factor, temp_range_scaled):
    #     intermediate_result = f_i_temps_old[:, :, np.newaxis] - window_biases
    #     exponential_term = np.exp(intermediate_result * beta_range[:, np.newaxis, np.newaxis])
    #     denominator = summed_p_i_b_s[:, :, np.newaxis] * exponential_term 
    
    #     p_x = numerator / np.sum(denominator, axis=1)
    #     sum_p_bf = np.sum(p_x[:, np.newaxis, :] * f_i_bias_factor, axis=2)
    #     f_i_temps_new = -temp_range_scaled[:, np.newaxis] * np.log(sum_p_bf)
    
    #     return f_i_temps_new, px
    
    def wham_cont_and_disc_temp_interp_converg_analysis(self, convergence_slice, temp_range, n_bins, xmin, xmax, umbrella_stiff, max_hb, epsilon=1e-7, reread_files=False, max_iterations=100000):
        self.wham_temp_interp_converg_analysis(convergence_slice, temp_range, n_bins, xmin, xmax, max_hb, umbrella_stiff, epsilon=epsilon, reread_files=reread_files, max_iterations=max_iterations)
        self.discrete_temp_interp_converg_analysis(convergence_slice, xmin, xmax, max_hb, temp_range, reread_files=reread_files)
    
    def discrete_temp_interp_converg_analysis(self, convergence_slice, xmin, xmax, max_hb, temp_range, reread_files=False):
        if (type(convergence_slice) == int) or (type(convergence_slice) == float):
            min_length = min([len(inner_list['com_distance']) for inner_list in self.obs_df])
            #split min_length into 3 slices
            convergence_slice = np.array_split(np.arange(min_length), convergence_slice)
            convergence_slice = [slice(slice_[0], slice_[-1], 1) for slice_ in convergence_slice]
        
        self.convergence_discrete_free_energy = []
        self.convergence_discrete_prob_discrete = []
        
        self.convergence_Tm = []
        self.convergence_x_fit = []
        self.convergence_y_fit = []
        self.convergence_inverted_finfs = []
        
        
        for idx, converg_slice in enumerate(convergence_slice):
            self.F_i = self.convergence_F_i_temps[idx]
            free_energy, prob_discrete = self.temperature_interpolation(xmin, xmax, max_hb, temp_range, reread_files=reread_files, convergence_slice=converg_slice)
            self.convergence_discrete_free_energy.append(free_energy)
            self.convergence_discrete_prob_discrete.append(prob_discrete)
            
            self.calculate_melting_temperature(temp_range)
            print(self.Tm)
            self.convergence_Tm.append(self.Tm)
            self.convergence_x_fit.append(self.x_fit)
            self.convergence_y_fit.append(self.y_fit)
            self.convergence_inverted_finfs.append(self.inverted_finfs)
            
    
    def wham_temp_interp_converg_analysis(self, convergence_slice, temp_range, n_bins, xmin, xmax, umbrella_stiff, max_hb, epsilon=1e-7, reread_files=False, all_observables=False, max_iterations=100000):
        if (type(convergence_slice) == int) or (type(convergence_slice) == float):
            min_length = min([len(inner_list['com_distance']) for inner_list in self.obs_df])
            #split min_length into 3 slices
            convergence_slice = np.array_split(np.arange(min_length), convergence_slice)
            convergence_slice = [slice(slice_[0], slice_[-1], 1) for slice_ in convergence_slice]
        
        
        self.convergence_free = []
        self.convergence_F_i_temps = []
        for converg_slice in convergence_slice:
            free, F_i_temps, f_i_temps_over_time = self.wham_temperature_interpolation(temp_range, n_bins, xmin, xmax, max_hb, umbrella_stiff,
                                                                                  epsilon=epsilon, reread_files=reread_files, max_iterations=max_iterations, convergence_slice=converg_slice)
            self.convergence_free.append(free)
            self.convergence_F_i_temps.append(F_i_temps)
            
    
    def wham_temperature_interpolation(self, temp_range, n_bins, xmin, xmax, umbrella_stiff, max_hb, epsilon=1e-7, reread_files=False, all_observables=False, max_iterations=100000, convergence_slice=None):
        
        if reread_files is False:
            if self.obs_df is None:
                self.analysis.read_all_observables('prod')
        elif reread_files is True:
            self.analysis.read_all_observables('prod')
        
        number_of_forces = self.get_n_external_forces()
        force_energy = [f'force_energy_{idx}' for idx in range(number_of_forces)]
        # force_energy = ['force_energy']

        min_length = min([len(inner_list['com_distance']) for inner_list in self.obs_df])
        truncated_com_values = [inner_list['com_distance'][:min_length] for inner_list in self.obs_df] 
        x_range = np.round(np.linspace(xmin, xmax, (self.n_windows + 1), dtype=np.double)[1:], 3)
        names = ['backbone', 'bonded_excluded_volume', 'stacking', 'nonbonded_excluded_volume', 'hydrogen_bonding', 'cross_stacking', 'coaxial_stacking', 'debye_huckel']
        truncated_potential_energy = [inner_list[names][:min_length] for inner_list in self.obs_df]
        truncated_kinetic_energy = [inner_list['kinetic_energy'][:min_length] for inner_list in self.obs_df]
        truncated_force_energy = [inner_list[force_energy][:min_length] for inner_list in self.obs_df]
        truncated_umbrella_bias = [0.5 * np.double(umbrella_stiff) * (com_values - eq_pos)**2 for com_values, eq_pos in zip(truncated_com_values, x_range)]
        
        
        if convergence_slice is not None:
            truncated_com_values = [inner_list[convergence_slice] for inner_list in truncated_com_values] 
            names = ['backbone', 'bonded_excluded_volume', 'stacking', 'nonbonded_excluded_volume', 'hydrogen_bonding', 'cross_stacking', 'coaxial_stacking', 'debye_huckel']
            truncated_potential_energy = [inner_list[names][convergence_slice] for inner_list in truncated_potential_energy]
            truncated_kinetic_energy = [inner_list[convergence_slice] for inner_list in truncated_kinetic_energy]
            truncated_force_energy = [inner_list[force_energy][convergence_slice] for inner_list in truncated_force_energy]
            truncated_umbrella_bias = [0.5 * np.double(umbrella_stiff) * (com_values - eq_pos)**2 for com_values, eq_pos in zip(truncated_com_values, x_range)]
        # w_i = self.weight_sample()


        #Get temperature scalar
        temp_range_scaled = self.celcius_to_scaled(temp_range)
        beta_range = 1 / temp_range_scaled

        self.get_temperature()
        temperature = np.array(self.temperature, dtype=np.double)
        beta = 1 / temperature


        top_file = self.production_sims[0].sim_files.top
        with open(top_file, 'r') as f:
            n_particles_in_system = np.double(f.readline().split(' ')[0])
        
        n_particles_in_op = max_hb * 2
        
        truncated_non_pot_energy = truncated_kinetic_energy + (np.array(truncated_umbrella_bias) * 2 / n_particles_in_system)

        new_energy_per_window = self._new_calcualte_bias_energy(truncated_non_pot_energy, temp_range, truncated_potential_energy=truncated_potential_energy)
        new_energy_per_window = np.array(new_energy_per_window) * n_particles_in_system

        new_energy_per_window_reshaped = np.array(new_energy_per_window).swapaxes(0,1)
        # temp_biases = np.exp(np.array(new_energy_per_window).swapaxes(0,1))# *beta_range[:,np.newaxis, np.newaxis])
        
        calculated_bin_centers, bin_edges = self.get_bins(xmin, xmax, n_bins=n_bins)

        self.r0 = np.round(np.linspace(xmin, xmax, (self.n_windows + 1))[1:], 3)
        #Calculate the biases in the windows
        window_biases = np.array([[
            self.w_i(bin_value, r0_value, umbrella_stiff)
            for bin_value in calculated_bin_centers
            ] 
            for r0_value in self.r0
        ], dtype=np.double)

        #Get the com values
        all_com_values = np.array(truncated_com_values)

        # Calculate the counts of each bin
        # count = [
        #     [
        #     np.histogram(com_values, bins=bin_edges, weights=t_bias)[0]
        #     for com_values, t_bias in zip(all_com_values, temp_bias)
        #     ] 
        #     for temp_bias in temp_biases
        # ]
        # print(count)
        
        # Initialize the 4D list
        weights_in_bins = [[[[] for _ in range(len(bin_edges) -1 )] for _ in range(len(self.obs_df)) ] for _ in range(len(temp_range))] 

        bin_idx = np.digitize(all_com_values, bin_edges) - 1  # -1 because np.digitize starts from 1
        # return bin_idx
        # Populate the weights in bins
        for temp_idx, temp_bias in enumerate(new_energy_per_window_reshaped):
            print(f'{temp_idx=}')
            for window_idx, (bin_i, t_bias) in enumerate(zip(bin_idx, temp_bias)):
                for val_idx, val in enumerate(bin_i):
                    # Find the bin index for this value
                    # bin_idx = np.digitize(val, bin_edges) - 1  # -1 because np.digitize starts from 1
                    if val < n_bins:
                        weights_in_bins[temp_idx][window_idx][val].append(t_bias[val_idx])
        # Convert to 3D array and apply logsumexp
        n_temps = len(temp_range)
        n_windows = len(self.obs_df)
        n_bins = len(bin_edges) - 1
        log_histogram = np.zeros((n_temps, n_windows, n_bins))
        for temp_idx in range(n_temps):
            for window_idx in range(n_windows):
                for bin_idx in range(n_bins):
                    if len(weights_in_bins[temp_idx][window_idx][bin_idx]) > 0:
                        log_histogram[temp_idx, window_idx, bin_idx] = logsumexp(weights_in_bins[temp_idx][window_idx][bin_idx])
                    else:
                        log_histogram[temp_idx, window_idx, bin_idx] = 0

        self.log_histogram = log_histogram
        
        
        # count = self.fast_histogram(all_com_values, temp_biases, bin_edges)

        #Put the counts into an array
        # p_i_b_s = np.array(count)
        # p_i_b_s = np.exp(log_histogram, out=np.zeros_like(log_histogram, dtype=np.double), where=log_histogram!=0)
        norm_factor = np.max(logsumexp(log_histogram, axis=2, keepdims=True), axis=1)
        p_i_b_s = np.exp(log_histogram - norm_factor[:, np.newaxis, :], out=np.zeros_like(log_histogram, dtype=np.double), where=log_histogram!=0)
        summed_p_i_b_s = np.sum(p_i_b_s, axis=2)
        
        # return p_i_b_s, log_histogram
        beta_range_reshaped = beta_range[:, np.newaxis, np.newaxis]
        
        # return p_i_b_s, log_histogram
        #The numerator of p_x is the sum of the counts from each window
        numerator = np.sum(p_i_b_s, axis=1)
        
        
        rng = np.random.default_rng()    
        # epsilon = 1e-7

        f_i_bias_factor = np.array([np.exp(-window_biases * bet) for bet in beta_range])
        f_i_temps_old = np.array([[rng.normal(loc=0.0, scale=1.0, size=None) for _ in range(len(self.obs_df))] for _ in temp_range_scaled])
        f_i_temps_new = np.zeros_like(f_i_temps_old)
        f_i_temps_over_time = []
        
        first = True
        iteration = 0
        update_frequency = 1000
        significant_digits = abs(int(np.floor(np.log10(abs(epsilon))))) +1
        custom_bar_format = '{desc} {r_bar}'
        with tqdm(desc='WHAM', leave=True, bar_format=custom_bar_format) as pbar:
            while ((first is True) or (np.max(np.abs(f_i_temps_new - f_i_temps_old)) > epsilon)) and (iteration < max_iterations):
                f_i_temps_old = deepcopy(f_i_temps_new)
                # print(f'{type(f_i_temps_old)=}{type(window_biases)=}{type(beta_range)=}{type(summed_p_i_b_s)=}{type(numerator)=}{type(f_i_bias_factor)=}{type(temp_range_scaled)=}{type(f_i_temps_new)=}{type(f_i_temps_over_time)=}')
                # f_i_temps_new = self.compute_f_i_temps(f_i_temps_old, window_biases, beta_range, summed_p_i_b_s, numerator, f_i_bias_factor, temp_range_scaled)
                intermediate_result = f_i_temps_old[:, :, np.newaxis] - window_biases
                exponential_term = np.exp(intermediate_result * beta_range[:, np.newaxis, np.newaxis])
                denominator = summed_p_i_b_s[:, :, np.newaxis] * exponential_term 
                                
                #Compute the probability of each bin
                p_x = numerator / np.sum(denominator, axis=1)

                #Recompute the f_i values per window. This value will update till convergence
                sum_p_bf = np.sum(p_x[:, np.newaxis, :] * f_i_bias_factor, axis=2)

                f_i_temps_new = -temp_range_scaled[:,np.newaxis] * np.log(sum_p_bf)

                convergence_criterion = np.max(np.abs(f_i_temps_new - f_i_temps_old))
                f_i_temps_over_time.append(convergence_criterion)
                iteration +=1

                if (iteration % update_frequency == 0) or (iteration == max_iterations) or (first == True):
                    formatted_convergence = f"{convergence_criterion:.{significant_digits}f} / {float(epsilon)}"
                    pbar.set_postfix_str(f"Convergence: {formatted_convergence}")
                    if first is True:
                        pbar.update(0)
                    else:
                        pbar.update(update_frequency)
                first = False

        value = f_i_temps_new[:,0]
        F_i_temps = f_i_temps_new - value[:,np.newaxis]
        # F_i_temps = F_i_temps * n_particles_in_system
        
        #Get free energy
        free = -np.log(p_x)
        # free = free * n_particles_in_system
        free = free - np.min(free, axis=1, keepdims=True)

        if iteration < max_iterations:
            print(f'Converged in [{iteration}] iterations')
        else:
            print(f'Failed to converge in [{iteration}] iterations')
            
        self.F_i = F_i_temps
        return free, F_i_temps, f_i_temps_over_time
    
    def w_i(self, r, r0, stiff):
        return 0.5 * float(stiff) * (r - r0)**2

    def get_bins(self, xmin, xmax, n_bins=200):
        # Calculate the bin width
        bin_width = (xmax - xmin) / n_bins

        # Calculate the first bin center
        first_bin_center = xmin + bin_width / 2

        # Generate the bin centers
        calculated_bin_centers = np.array([first_bin_center + i * bin_width for i in range(n_bins)])

        # Calculate bin edges based on bin centers
        bin_edges = np.zeros(len(calculated_bin_centers) + 1)
        bin_edges[1:-1] = (calculated_bin_centers[:-1] + calculated_bin_centers[1:]) / 2.0
        bin_edges[0] = calculated_bin_centers[0] - (bin_edges[1] - calculated_bin_centers[0])
        bin_edges[-1] = calculated_bin_centers[-1] + (calculated_bin_centers[-1] - bin_edges[-2])

        return calculated_bin_centers, bin_edges

    def celcius_to_scaled(self, temp):
        return (temp + 273.15) / 3000
    
    def weight_sample(self, epsilon=1e-15, reread_files=False, convergence_slice=None):

        if reread_files is False:
            if self.obs_df is None:
                self.analysis.read_all_observables('prod')
        elif reread_files is True:
            self.analysis.read_all_observables('prod')
             
        number_of_forces = self.get_n_external_forces()
        force_energy = [f'force_energy_{idx}' for idx in range(number_of_forces)]


        min_length = min([len(inner_list['com_distance']) for inner_list in self.obs_df])        
        truncated_force_energy = [inner_list[force_energy][:min_length] for inner_list in self.obs_df]
        
        
        if convergence_slice is not None:
            truncated_force_energy = [inner_list[convergence_slice] for inner_list in truncated_force_energy]

        temperature = np.array(self.temperature, dtype=np.double)
        beta = 1 / temperature
        
        
        truncated_com_distance = pd.concat([inner_list['com_distance'][:min_length] for inner_list in self.obs_df]).reset_index(drop=True)
        r0_list = self.r0

        com_grid, r0_grid = np.meshgrid(truncated_com_distance, r0_list)
        
        def force_potential_energy(r0, com_position):
            return 0.5 * 5 * (com_position - r0)**2
        result = force_potential_energy(r0_grid, com_grid)

        number_of_forces = self.get_n_external_forces()
        force_energy = [f'force_energy_{idx}' for idx in range(number_of_forces)]


        force_samples = np.concatenate([result, np.array(np.concatenate(truncated_force_energy))[:, :16].T])
        force_samples =  force_samples.T 
        force_samples[:, 17:] = force_samples[:, 17:] / 16
        
        # force_samples = pd.concat(truncated_force_energy).reset_index(drop=True)
        
        rng = np.random.default_rng()    
        f_i_old = np.array([rng.normal(loc=0.0, scale=1.0, size=None) for _ in range(force_samples.shape[1])])
        
        
        top_file = self.production_sims[0].sim_files.top
        with open(top_file, 'r') as f:
            n_particles_in_system = np.double(f.readline().split(' ')[0])
        
        e_to_neg_u_beta = np.exp(-force_samples * n_particles_in_system * beta)
        e_to_neg_fi_beta_old = np.exp(-f_i_old * beta)
        e_to_neg_fi_beta_new = np.zeros_like(e_to_neg_fi_beta_old)

        convergence_criterion_list = []
        convergence_criterion = np.max(np.abs(e_to_neg_fi_beta_new - e_to_neg_fi_beta_old))

        while convergence_criterion > epsilon:
            denom = np.sum(e_to_neg_u_beta * (1 / e_to_neg_fi_beta_old), axis=1)

            numerator = 1 / np.sum(1 / denom)

            w_i =  numerator / denom

            e_to_neg_fi_beta_new = np.sum(w_i[:, np.newaxis] * e_to_neg_u_beta, axis=0)

            convergence_criterion = np.max(np.abs(e_to_neg_fi_beta_new - e_to_neg_fi_beta_old))
            print(convergence_criterion)

            convergence_criterion_list.append(convergence_criterion)
            e_to_neg_fi_beta_old = deepcopy(e_to_neg_fi_beta_new)
        
        
        return w_i
    
    
    def read_kinetic_and_potential_energy(self):
        self.energy_by_window = {}
        for idx,sim in enumerate(self.production_sims):
            sim.sim_files.parse_current_files()
            
            # Read the entire file into a DataFrame
            df = pd.read_csv(sim.sim_files.energy, delim_whitespace=True,names=['time', 'U','P','K'])
            self.energy_by_window[idx] = df
    
    def read_potential_energy(self):
        self.potential_energy_by_window = {}
        names = ['backbone', 'bonded_excluded_volume', 'stacking', 'nonbonded_excluded_volume', 'hydrogen_bonding', 'cross_stacking', 'coaxial_stacking', 'debye_huckel']
        
        for idx, sim in enumerate(self.production_sims):
            sim.sim_files.parse_current_files()
            
            # Read the entire file into a DataFrame
            df = pd.read_csv(sim.sim_files.potential_energy, header=None, names=names, delim_whitespace=True, dtype=np.double)
            
            self.potential_energy_by_window[idx] = df
            
    def read_hb_contacts(self, sim_type='prod'):
        if sim_type == 'prod':
            sim_list = self.production_sims
        elif sim_type == 'eq':
            sim_list = self.equlibration_sims
        elif sim_type == 'pre_eq':
            sim_list = self.pre_equlibration_sims
            
        self.hb_contacts_by_window = {}
        for idx,sim in enumerate(sim_list):
            sim.sim_files.parse_current_files()
            df = pd.read_csv(sim.sim_files.hb_contacts, header=None, engine='pyarrow')
            self.hb_contacts_by_window[idx] = df

    def get_hb_list_by_window(self):
        hb_list_by_window = {}
        for idx,sim in enumerate(self.production_sims):
            sim.sim_files.parse_current_files()
            df = pd.read_csv(sim.sim_files.hb_observable, header=None, engine='pyarrow')
            hb_list_by_window[idx] = df
        self.hb_by_window = hb_list_by_window
        
    def write_potential_energy_files(self):
        all_observables = self.analysis.read_all_observables('prod')
        names = ['backbone', 'bonded_excluded_volume', 'stacking', 'nonbonded_excluded_volume', 'hydrogen_bonding', 'cross_stacking', 'coaxial_stacking', 'debye_huckel']
        sim_dirs = [sim.sim_dir for sim in self.production_sims]
        for df, sim_dir in zip(all_observables, sim_dirs):
            potential_energy_terms = df[names].values
            with open(os.path.join(sim_dir, 'potential_energy.txt'), 'w') as f:
                for row in potential_energy_terms:
                    row_str = ' '.join(map(str, row))
                    f.write(row_str + '\n')

        return None
    
    def copy_hb_list_to_com_dir(self):
        copy_h_bond_files(self.production_sim_dir, self.com_dir)

    def spawn_continuous_to_discrete_unbiasing(self, max_hb, join=False):
        self.spawn(self.continuous_to_discrete_unbiasing, args=(max_hb,), join=False)
    
    def continuous_to_discrete_unbiasing(self, max_hb):
        def count_division_normalize(arr):
            row_sums = np.sum(arr, axis=1, keepdims=True, dtype=np.double)
            return np.divide(arr, row_sums, out=np.zeros_like(arr, dtype=np.double), where=row_sums!=0)
        
        if self.com_by_window is None:
            self.get_com_distance_by_window()
        if self.hb_by_window is None:
            self.get_hb_list_by_window()
        if self.umbrella_bias is None:
            self.get_bias_potential_value(self.wham.xmin, self.wham.xmax, self.n_windows, self.wham.umbrella_stiff)
                
        unbiased_discrete_window = np.array([np.zeros(max_hb + 1) for _ in range(self.n_windows)], dtype=np.double)

        temperature = np.array(self.temperature, dtype=np.double)
        beta = 1 / temperature
        bias = []
        bias_norm = []
        for idx, window in enumerate(self.umbrella_bias):
            bias_values = window.values.T[0]
            bias.append(beta * bias_values)
            bias_norm.append(beta * bias_values)

        self.bias = bias
        index_to_add_at = [np.array(self.hb_by_window[idx].values.T[0]) for idx in range(self.n_windows)]
        
        
        hb_by_window = np.array(list(self.hb_by_window.values())).squeeze(-1)
        hb_by_window = np.where(hb_by_window <= max_hb, hb_by_window, max_hb)
        index_to_add_at = hb_by_window
        self.index_to_add_at = index_to_add_at

        b_u_sq_hb_list = [[[] for _ in range(max_hb + 1)] for _ in range(self.n_windows)]        

        for win_idx, (b_u_win,hb_win) in enumerate(zip(bias, index_to_add_at)):
            for b_u, hb in zip(b_u_win, hb_win):
                b_u_sq_hb_list[win_idx][hb].append(b_u)

        e_to_beta_u_sq = np.empty((self.n_windows, max_hb+1), dtype=np.double)

        for win_idx, b_u_hb_lists in enumerate(b_u_sq_hb_list):
            for hb_idx, b_u_hb in enumerate(b_u_hb_lists):
                if len(b_u_hb) > 0:
                    e_to_beta_u_sq[win_idx][hb_idx] = logsumexp(b_u_hb)
                else:
                    e_to_beta_u_sq[win_idx][hb_idx] = 0

        e_to_beta_u = [logsumexp(b_U) for b_U in bias]
        self.e_to_beta_u = e_to_beta_u
        self.e_to_beta_u_sq = e_to_beta_u_sq
        
        log_p_i_h = [e_b_u_sq - e_b_u for e_b_u_sq,e_b_u in zip(e_to_beta_u_sq, e_to_beta_u)] 
        self.log_p_i_h = log_p_i_h

        
        f_i = self.get_biases()
        weight = -beta * np.array(f_i)
        self.weight = weight
        weight_norm = logsumexp(weight)
        A_i = weight - weight_norm

        self.com_max = np.max(np.array([com_dist for com_dist in self.com_by_window.values()]))
        last_conf_file = self.production_sims[0].sim_files.last_conf
        with open(last_conf_file, 'r') as f:
            next(f)
            box_info = f.readline().split(' ')
            self.box_size = float(box_info[-1].strip())
        
        self.volume_correction = np.log((self.box_size**3) / ((4/3)*np.pi*self.com_max**3))
        
        self.windowed_log_prob_discrete = np.array([p_i + a for p_i,a in zip(log_p_i_h, A_i)])
        self.log_prob_discrete = np.array([logsumexp(hb_list) for hb_list in self.windowed_log_prob_discrete.T])

        self.free_energy_discrete = np.array([-free for free in self.log_prob_discrete])
        self.free_energy_discrete -= self.free_energy_discrete[0]
        self.free_energy_discrete[0] -= self.volume_correction
        self.free_energy_discrete += self.volume_correction
        
        self.normed_free_energy = -self.free_energy_discrete - logsumexp(-self.free_energy_discrete)
        
        self.prob_discrete = np.exp(self.normed_free_energy)
        
    def modify_topology_for_unique_pairing(self):
        """
        Modify the topology file to ensure that each nucleotide can only bind to its original partner.
        """
        for sim in self.equlibration_sims:
            topology_file_path = sim.sim_files.top  # Assuming this is the absolute path to the topology file
            
            # Read the existing topology file
            with open(topology_file_path, 'r') as f:
                lines = f.readlines()
            
            # Initialize variables
            new_lines = []
            next_available_base_type = 13  # Start from 13 as per your example
            
            # Process the header
            new_lines.append(lines[0])  # Keep the header as is
            
            num_base_pairs = len(lines[1:])
            bp_per_strand = num_base_pairs // 2
            if int(num_base_pairs / bp_per_strand) != 2:
                return print('Even number of base pairs required')
            
            # Process the first strand
            for line in lines[1:bp_per_strand +1]:  
                parts = line.split()
                strand_id, base_type, prev_idx, next_idx = parts
                
                # Modify the base type to ensure unique pairing
                unique_base_type = next_available_base_type
                next_available_base_type += 1  # Increment for the next base
                
                new_line = f"{strand_id} {unique_base_type} {prev_idx} {next_idx}\n"
                new_lines.append(new_line)
            
            second_strand_base_type = -next_available_base_type + 4
            # Generate the complementary strand with negative unique base types
            for line in lines[bp_per_strand+1:]:
                parts = line.split()
                strand_id, base_type, prev_idx, next_idx = parts
                
                # Modify the base type for the complementary strand
                unique_base_type = int(second_strand_base_type)
                second_strand_base_type += 1
                
                new_line = f"{strand_id} {unique_base_type} {prev_idx} {next_idx}\n"
                new_lines.append(new_line)
            
            # Write the modified topology back to the file
            with open(topology_file_path, 'w') as f:
                f.writelines(new_lines) 
    
class NDimensionalUmbrella(MeltingUmbrellaSampling):
    def __init__(self, file_dir, production_sim_dir):
        super().__init__(file_dir, production_sim_dir)
        
    #I want to create an n-dimensional umbrella sampling class
    #In order to run n-dimensional umbrella sampling I need to have a seperate force for each dimension
    #In 3-D umbrella sampling I will have the first force be 10 diffrent values, then for each one of those 10 values
    #I will have 10 values for the second force, and then for each of those 10 values I will have 10 values for the third force
    #This will give me 1000 windows
    #I wonder if there is a way to simplify this process
    #Well either way for now I need to just implment the simple version
    #The first step in setting this up is thinking about how I will generate the windows
    
    #I can create a simulation object for each window of course
    #The only diffrence between each simulation window is that the force is diffrent
    #This means I need a function that will generate the force for each window
    #I will also need a diffrent com_distance observable for each dimension
    
    def build_equlibration_runs(self, simulation_manager, n_dimensions, n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                                observable=False, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False,
                                protein=None, force_file=None):
        self.observables_list = []
        self.windows.equlibration_windows(n_windows)
        
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        
        
        self.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        self.hb_list_observable(print_every=print_every, only_count='true', name=name)
        self.force_energy_observable(print_every=print_every, name=name)
        self.potential_energy_observable(print_every=print_every, name=name)
        
        if continue_run is False:
            self.us_build.build(self.equlibration_sims, input_parameters,
                                self.forces_list, self.observables_list,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file)
            for sim in self.equlibration_sims:
                sim.build_sim.build_hb_list_file(com_list, ref_list)
        self.queue_sims(simulation_manager, self.equlibration_sims, continue_run=continue_run)
    
    def com_distance_observable(self, com_list, ref_list,  print_every=1e4, name='com_distance.txt'):
        """ Build center of mass observable"""
        obs = Observable()
        com_observable = obs.distance(
            particle_1=com_list,
            particle_2=ref_list,
            print_every=f'{print_every}',
            name=f'{name}',
            PBC='1'
        )  
        self.observables_list.append(com_observable)
 

    def umbrella_forces(self, com_list, ref_list, stiff, xmin, xmax, n_windows):
        """ Build Umbrella potentials"""
        x_range = np.round(np.linspace(xmin, xmax, (n_windows + 1))[1:], 3)
        umbrella_forces_1 = []
        umbrella_forces_2 = []
        
        for x_val in x_range:   
            self.umbrella_force_1 = self.f.com_force(
                com_list=com_list,                        
                ref_list=ref_list,                        
                stiff=f'{stiff}',                    
                r0=f'{x_val}',                       
                PBC='1',                         
                rate='0',
            )        
            umbrella_forces_1.append(self.umbrella_force_1)
            
            self.umbrella_force_2= self.f.com_force(
                com_list=ref_list,                        
                ref_list=com_list,                        
                stiff=f'{stiff}',                    
                r0=f'{x_val}',                       
                PBC='1',                          
                rate='0',
            )
            umbrella_forces_2.append(self.umbrella_force_2)  
        self.forces_list = np.transpose(np.array([umbrella_forces_1, umbrella_forces_2]))     

    def rate_umbrella_forces(self, com_list, ref_list, stiff, xmin, xmax, n_windows, starting_r0, steps):
        """ Build Umbrella potentials"""
        
        x_range = np.round(np.linspace(xmin, xmax, (n_windows + 1))[1:], 3)
        
        umbrella_forces_1 = []
        umbrella_forces_2 = []
        
        for x_val in x_range:   
            force_rate_for_x_val = (x_val - starting_r0) / steps
            
            self.umbrella_force_1 = self.f.com_force(
                com_list=com_list,                        
                ref_list=ref_list,                        
                stiff=f'{stiff}',                    
                r0=f'{starting_r0}',                       
                PBC='1',                         
                rate=f'{force_rate_for_x_val}',
            )        
            umbrella_forces_1.append(self.umbrella_force_1)
            
            self.umbrella_force_2= self.f.com_force(
                com_list=ref_list,                        
                ref_list=com_list,                        
                stiff=f'{stiff}',                    
                r0=f'{starting_r0}',                       
                PBC='1',                          
                rate=f'{force_rate_for_x_val}',
            )
            umbrella_forces_2.append(self.umbrella_force_2)  
        self.forces_list = np.transpose(np.array([umbrella_forces_1, umbrella_forces_2])) 
    
    
class UmbrellaAnalysis:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
        self.base_umbrella.obs_df = None
    
    def read_all_observables(self, sim_type):
        

        file_name = self.base_umbrella.observables_list[0]['output']['name']
        print_every = int(float(self.base_umbrella.observables_list[0]['output']['print_every']))
        
        if sim_type == 'eq':
            sim_list = self.base_umbrella.equlibration_sims
        elif sim_type == 'prod':
            sim_list = self.base_umbrella.production_sims
        elif sim_type == 'pre_eq':
            sim_list = self.base_umbrella.pre_equlibration_sims
        
        obs_types = [observe['output']['cols'][0]['type'] for observe in self.base_umbrella.observables_list]


        with open(sim_list[0].sim_files.force, 'r') as f:
            force_js = load(f)
            number_of_forces = len(force_js.keys())

                        
        all_observables = []
        for sim in sim_list:
            try:
                all_observables.append(pd.read_csv(f"{sim.sim_dir}/{file_name}", header=None, engine='pyarrow'))
            except FileNotFoundError:
                all_observables.append(pd.DataFrame())
        
        names = ['backbone', 'bonded_excluded_volume', 'stacking', 'nonbonded_excluded_volume', 'hydrogen_bonding', 'cross_stacking', 'coaxial_stacking', 'debye_huckel']
        force_energy = [f'force_energy_{idx}' for idx in range(number_of_forces)]
        columns = ['com_distance', 'hb_list', *force_energy, 'kinetic_energy', *names]
        # columns = ['com_distance', 'hb_list', 'force_energy', 'kinetic_energy', *names]

        
        obs = [pd.DataFrame([
                list(filter(lambda a: a != '',all_observables[window_idx].iloc[data_idx][0].split(' ')))
                for data_idx in range(len(all_observables[window_idx]))
                ],columns=columns, dtype=np.double)
                for window_idx in range(len(all_observables))
            ]
        
        obs = [pd.concat([obs[window_idx],
                  pd.DataFrame([np.arange(len(obs[window_idx])) * print_every]
                           , dtype=np.int64).T.rename(columns={0: 'steps'})]
                         , axis=1)
               for window_idx in range(len(obs))]
        
        self.base_umbrella.obs_df = obs
        return self.base_umbrella.obs_df
        
    def view_observable(self, sim_type, idx, sliding_window=False, observable=None):
        if observable == None:
            observable=self.base_umbrella.observables_list[0]
        
        if sim_type == 'eq':
            self.base_umbrella.equlibration_sims[idx].analysis.plot_observable(observable, fig=False, sliding_window=sliding_window)

        if sim_type == 'prod':
            self.base_umbrella.production_sims[idx].analysis.plot_observable(observable, fig=False, sliding_window=sliding_window)
    
    def hist_observable(self, sim_type, idx, bins=10, observable=None):
        if observable == None:
            observable=self.base_umbrella.observables_list[0]
            
        if sim_type == 'eq':
            self.base_umbrella.equlibration_sims[idx].analysis.hist_observable(observable,
                                                                               fig=False, bins=bins)

        if sim_type == 'prod':
            self.base_umbrella.production_sims[idx].analysis.hist_observable(observable,
                                                                             fig=False, bins=bins)
            
    def hist_cms(self, sim_type, idx, xmax, print_every, bins=10, fig=True):
        if sim_type == 'eq':
            self.base_umbrella.equlibration_sims[idx].analysis.hist_cms_obs(xmax, print_every, bins=bins, fig=fig)
        if sim_type == 'prod':
            self.base_umbrella.production_sims[idx].analysis.hist_cms_obs(xmax, print_every,bins=bins, fig=fig)
          
    def view_cms(self, sim_type, idx, xmax, print_every, sliding_window=10, fig=True):
        if sim_type == 'eq':
            self.base_umbrella.equlibration_sims[idx].analysis.view_cms_obs(xmax, print_every, sliding_window=sliding_window, fig=fig)
        if sim_type == 'prod':
            self.base_umbrella.production_sims[idx].analysis.view_cms_obs(xmax, print_every, sliding_window=sliding_window, fig=fig)
  
    def combine_hist_observable(self, observable, idxes, bins=10, fig=True):
        for sim in self.base_umbrella.production_sims[idx]:    
            file_name = observable['output']['name']
            conf_interval = float(observable['output']['print_every'])
            df = pd.read_csv(f"{self.sim.sim_dir}/{file_name}", header=None, engine='pyarrow')
            df = np.concatenate(np.array(df))
            H, bins = np.histogram(df, density=True, bins=bins)
            H = H * (bins[1] - bins[0])
    
    def view_observables(self, sim_type, sliding_window=False, observable=None):
        if observable == None:
            observable=self.base_umbrella.observables_list[0]
            
        if sim_type == 'eq':
            plt.figure()
            for sim in self.base_umbrella.equlibration_sims:
                sim.analysis.plot_observable(observable, fig=False, sliding_window=sliding_window)

        if sim_type == 'prod':
            plt.figure(figsize=(15,3))
            for sim in self.base_umbrella.production_sims:
                sim.analysis.plot_observable(observable, fig=False, sliding_window=sliding_window)
    
    def view_conf(self, sim_type, window):
        if sim_type == 'eq':
            try:
                self.base_umbrella.equlibration_sims[window].analysis.view_last()
            except:
                self.base_umbrella.equlibration_sims[window].analysis.view_init()
        if sim_type == 'prod':
            try:
                self.base_umbrella.production_sims[window].analysis.view_last()
            except:
                self.base_umbrella.production_sims[window].analysis.view_init()    
    
        
        
class UmbrellaWindow:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
    
    def pre_equlibration_windows(self, n_windows):
        self.base_umbrella.pre_equlibration_sim_dir = join(self.base_umbrella.system_dir, 'pre_equlibration')
        if not exists(self.base_umbrella.pre_equlibration_sim_dir):
            os.mkdir(self.base_umbrella.pre_equlibration_sim_dir)
        self.base_umbrella.pre_equlibration_sims = [Simulation(self.base_umbrella.file_dir, join(self.base_umbrella.pre_equlibration_sim_dir, str(window))) for window in range(n_windows)]
    
    def equlibration_windows(self, n_windows):
        """ 
        Sets a attribute called equlibration_sims containing simulation objects for all equlibration windows.
        
        Parameters:
            n_windows (int): Number of umbrella sampling windows.
        """
        self.base_umbrella.n_windows = n_windows
        self.base_umbrella.equlibration_sim_dir = join(self.base_umbrella.system_dir, 'equlibration')
        if not exists(self.base_umbrella.equlibration_sim_dir):
            os.mkdir(self.base_umbrella.equlibration_sim_dir)
        
        if hasattr(self.base_umbrella, 'pre_equlibration_sim_dir'):
            self.base_umbrella.equlibration_sims = [Simulation(sim.sim_dir, join(self.base_umbrella.equlibration_sim_dir, sim.sim_dir.split('/')[-1])) for sim in self.base_umbrella.pre_equlibration_sims]
        
        else:
            self.base_umbrella.equlibration_sims = [Simulation(self.base_umbrella.file_dir,join(self.base_umbrella.equlibration_sim_dir, str(window))) for window in range(n_windows)]
     
    
    def production_windows(self, n_windows):
        """ 
        Sets a attribute called production_sims containing simulation objects for all production windows.
        
        Parameters:
            n_windows (int): Number of umbrella sampling windows.
        """
        self.base_umbrella.production_sim_dir = join(self.base_umbrella.system_dir, 'production')
        if not exists(self.base_umbrella.production_sim_dir):
            os.mkdir(self.base_umbrella.production_sim_dir)
        self.base_umbrella.production_window_dirs = [join(self.base_umbrella.production_sim_dir, str(window)) for window in range(n_windows)]
        self.base_umbrella.production_sims = []
        for s, window_dir, window in zip(self.base_umbrella.equlibration_sims, self.base_umbrella.production_window_dirs, range(n_windows)):
            self.base_umbrella.production_sims.append(Simulation(self.base_umbrella.equlibration_sims[window].sim_dir, str(window_dir)))
    


class WhamAnalysis:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
    
    
    
    def run_wham(self, wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot):
        """
        Run Weighted Histogram Analysis Method on production windows.
        
        Parameters:
            wham_dir (str): Path to wham executable.
            xmin (str): Minimum distance of center of mass order parameter in simulation units.
            xmax (str): Maximum distance of center of mass order parameter in simulation units.
            umbrella_stiff (str): The parameter used to modified the stiffness of the center of mass spring potential
            n_bins (str): number of histogram bins to use.
            tol (str): Convergence tolerance for the WHAM calculations.
            n_boot (str): Number of monte carlo bootstrapping error analysis iterations to preform.

        """

        
        self.base_umbrella.com_dir = join(self.base_umbrella.production_sim_dir, 'com_dir')
        pre_temp = self.base_umbrella.production_sims[0].input.input['T']
        if ('C'.upper() in pre_temp) or ('C'.lower() in pre_temp):
            self.base_umbrella.temperature = (float(pre_temp[:-1]) + 273.15) / 3000
        elif ('K'.upper() in pre_temp) or ('K'.lower() in pre_temp):
             self.base_umbrella.temperature = float(pre_temp[:-1]) / 3000
        wham_analysis(wham_dir,
                      self.base_umbrella.production_sim_dir,
                      self.base_umbrella.com_dir,
                      str(xmin),
                      str(xmax),
                      str(umbrella_stiff),
                      str(n_bins),
                      str(tol),
                      str(n_boot),
                      str(self.base_umbrella.temperature))
        
        self.get_n_data_per_com_file()

    def get_n_data_per_com_file(self):
        com_dist_file = [file for file in os.listdir(self.base_umbrella.com_dir) if 'com_distance' in file][0]
        first_com_distance_file_name = join(self.base_umbrella.com_dir, com_dist_file)
        with open(first_com_distance_file_name, 'rb') as f:
            try:  # catch OSError in case of a one line file 
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
        self.base_umbrella.n_data_per_com_file = int(last_line.split()[0])
    
    def chunk_convergence_analysis(self, n_chunks):
        """
        Seperate your data into equal chunks
        """
        chunk_size = (self.base_umbrella.n_data_per_com_file // n_chunks)
        chunk_ends = [chunk_size * n_chunk for n_chunk in range(n_chunks + 1)]
        
        for idx, chunk in enumerate(chunk_ends):
            if chunk == 0:
                pass
            else:
                chunk_dir = join(self.base_umbrella.chunk_convergence_analysis_dir, f'{chunk_ends[idx - 1]}_{chunk}')
                if not exists(chunk_dir):
                    os.mkdir(chunk_dir)
        
        print(chunk_ends)
        
        self.chunk_dirs = []
        
        for idx, chunk in enumerate(chunk_ends):
            chunk_lower_bound = chunk_ends[idx - 1]
            chunk_upper_bound = chunk
            if chunk == 0:
                pass
            else:
                chunk_dir = join(self.base_umbrella.chunk_convergence_analysis_dir, f'{chunk_ends[idx - 1]}_{chunk}')
                self.chunk_dirs.append(chunk_dir)
                chunked_wham_analysis(chunk_lower_bound, chunk_upper_bound,
                                      self.wham_dir,
                                      self.base_umbrella.production_sim_dir,
                                      chunk_dir,
                                      str(self.xmin),
                                      str(self.xmax),
                                      str(self.umbrella_stiff),
                                      str(self.n_bins),
                                      str(self.tol),
                                      str(self.n_boot),
                                      str(self.base_umbrella.temperature))
                        
        return print(f'chunk convergence analysis')
    
    def data_truncated_convergence_analysis(self, data_added_per_iteration):
        """
        Seperate your data into equal chunks
        """
        chunk_size = (self.base_umbrella.n_data_per_com_file // data_added_per_iteration)
        chunk_ends = [chunk_size * n_chunk for n_chunk in range(data_added_per_iteration + 1)]
        
        for idx, chunk in enumerate(chunk_ends):
            if chunk == 0:
                pass
            else:
                chunk_dir = join(self.base_umbrella.data_truncated_convergence_analysis_dir, f'0_{chunk}')
                if not exists(chunk_dir):
                    os.mkdir(chunk_dir)
        
        print(chunk_ends)
        
        self.data_truncated_dirs = []
        
        for idx, chunk in enumerate(chunk_ends):
            chunk_lower_bound = 0
            chunk_upper_bound = chunk
            if chunk == 0:
                pass
            else:
                chunk_dir = join(self.base_umbrella.data_truncated_convergence_analysis_dir, f'0_{chunk}')
                self.data_truncated_dirs.append(chunk_dir)
                chunked_wham_analysis(chunk_lower_bound, chunk_upper_bound,
                                      self.wham_dir,
                                      self.base_umbrella.production_sim_dir,
                                      chunk_dir,
                                      str(self.xmin),
                                      str(self.xmax),
                                      str(self.umbrella_stiff),
                                      str(self.n_bins),
                                      str(self.tol),
                                      str(self.n_boot),
                                      str(self.base_umbrella.temperature))
                        
        return print(f'chunk convergence analysis')  
    
    
    def to_si(self, n_bins, com_dir):
        pre_temp = self.base_umbrella.production_sims[0].input.input['T']
        if ('C'.upper() in pre_temp) or ('C'.lower() in pre_temp):
            self.base_umbrella.temperature = (float(pre_temp[:-1]) + 273.15) / 3000
        elif ('K'.upper() in pre_temp) or ('K'.lower() in pre_temp):
             self.base_umbrella.temperature = float(pre_temp[:-1]) / 3000
        free = pd.read_csv(f'{com_dir}/freefile', sep='\t', nrows=int(n_bins))
        free['Free'] = free['Free'].div(self.base_umbrella.temperature)
        free['+/-'] = free['+/-'].div(self.base_umbrella.temperature)
        free['#Coor'] *= 0.8518
        return free     
    
    
    def w_mean(self, free_energy):
        free = free_energy.loc[:, 'Free']
        coord = free_energy.loc[:, '#Coor']
        prob = np.exp(-free) / sum(np.exp(-free))
        mean = sum(coord * prob)
        return mean
    
    
    def bootstrap_w_mean_error(self, free_energy, confidence_level=0.99):
        coord = free_energy.loc[:, '#Coor']
        free = free_energy.loc[:, 'Free'] 
        prob = np.exp(-free) / sum(np.exp(-free))
    
        err = free_energy.loc[:, '+/-']
        mask = np.isnan(err)
        err[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), err[~mask])
        cov = np.diag(err**2)
    
        estimate = np.array(multivariate_normal.rvs(mean=free, cov=cov, size=10000, random_state=None))
        est_prob = [np.exp(-est) / sum(np.exp(-est)) for est in estimate]
        means = [sum(coord * e_prob) for e_prob in est_prob]
        standard_error = np.std(means)
        z_score = norm.ppf(1 - (1 - confidence_level) / 2)
        confidence_interval = z_score * (standard_error / np.sqrt(len(means)))
        return standard_error, confidence_interval
        
    def plt_fig(self, title='Free Energy Profile', xlabel='End-to-End Distance (nm)', ylabel='Free Energy / k$_B$T'):
        from matplotlib.ticker import MultipleLocator
        with plt.style.context(['science', 'no-latex', 'bright']):
            plt.figure(dpi=200, figsize=(5.5, 4.5))
            # plt.title(title)
            plt.xlabel(xlabel, size=12)
            plt.ylabel(ylabel, size=12)
            #plt.rcParams['text.usetex'] = True
            # plt.rcParams['xtick.direction'] = 'in'
            # plt.rcParams['ytick.direction'] = 'in'
            # plt.rcParams['xtick.major.size'] = 6
            # plt.rcParams['xtick.minor.size'] = 4
            # plt.rcParams['ytick.major.size'] = 6
            # #plt.rcParams['ytick.minor.size'] = 4
            # plt.rcParams['axes.linewidth'] = 1.25
            # plt.rcParams['mathtext.fontset'] = 'stix'
            # plt.rcParams['font.family'] = 'STIXGeneral'
            ax = plt.gca()
            # ax.set_aspect('auto')
            # ax.xaxis.set_minor_locator(MultipleLocator(5))
            #ax.yaxis.set_minor_locator(MultipleLocator(2.5))
            # ax.tick_params(axis='both', which='major', labelsize=9)
            # ax.tick_params(axis='both', which='minor', labelsize=9)
            ax.yaxis.set_ticks_position('both')
        return ax

    
    def plot_indicator(self, indicator, ax, c='black', label=None):
        target = indicator[0]
        nearest = self.base_umbrella.free.iloc[(self.base_umbrella.free['#Coor'] -target).abs().argsort()[:1]]
        near_true = nearest
        x_val = near_true['#Coor']
        y_val = near_true['Free']
        ax.scatter(x_val, y_val, s=50)
        return None
    
    def plot_chunks_free_energy(self, ax=None, title='Free Energy Profile', label=None,errorevery=1):
        if ax is None:
            ax = self.plt_fig()
        for idx, df in enumerate(self.base_umbrella.chunk_dirs_free):
            # if label is None:
            label = self.chunk_dirs[idx].split('/')[-1]

            indicator = [self.base_umbrella.chunk_dirs_mean[idx], self.base_umbrella.chunk_dirs_standard_error[idx]]
            try:
                ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Free'],
                         yerr=df.loc[:, '+/-'], capsize=2.5, capthick=1.2,
                         linewidth=1.5, errorevery=errorevery, label=f'{label} {indicator[0]:.2f} nm \u00B1 {indicator[1]:.2f} nm')
                if indicator is not None:
                    self.plot_indicator(indicator, ax, label=label)
            except:
                ax.plot(df.loc[:, '#Coor'], df.loc[:, 'Free'], label=label)  
                
    def plot_truncated_free_energy(self, ax=None, title='Free Energy Profile', label=None,errorevery=1):
        if ax is None:
            ax = self.plt_fig()
        for idx, df in enumerate(self.base_umbrella.data_truncated_free):
            # if label is None:
            label = self.data_truncated_dirs[idx].split('/')[-1]

            indicator = [self.base_umbrella.data_truncated_mean[idx], self.base_umbrella.data_truncated_standard_error[idx]]
            try:
                ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Free'],
                         yerr=df.loc[:, '+/-'], capsize=2.5, capthick=1.2,
                         linewidth=1.5, errorevery=errorevery, label=f'{label} {indicator[0]:.2f} nm \u00B1 {indicator[1]:.2f} nm')
                if indicator is not None:
                    self.plot_indicator(indicator, ax, label=label)
            except:
                ax.plot(df.loc[:, '#Coor'], df.loc[:, 'Free'], label=label)  
    
    def plot_free_energy(self, ax=None, title='Free Energy Profile', label=None, errorevery=1, confidence_level=0.95):
        if ax is None:
            ax = self.plt_fig()
        if label is None:
            label = self.base_umbrella.system
    
        df = self.base_umbrella.free
        try:
            # Calculate the Z-value from the confidence level
            z_value = norm.ppf(1 - (1 - confidence_level) / 2)
    
            # Calculate the confidence interval
            confidence_interval = z_value * self.base_umbrella.standard_error
            ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Free'],
                        yerr=confidence_interval,  # Use confidence_interval here
                        capsize=2.5, capthick=1.2,
                        linewidth=1.5, errorevery=errorevery,
                        label=f'{label} {self.base_umbrella.mean:.2f} nm \u00B1 {confidence_interval:.2f} nm')
            if self.base_umbrella.mean is not None:
                self.plot_indicator([self.base_umbrella.mean, confidence_interval], ax, label=label)
        except:
            ax.plot(df.loc[:, '#Coor'], df.loc[:, 'Free'], label=label)
  
        

    def prob_plot_indicator(self, indicator, ax, label=None):
        target = indicator[0]
        nearest = self.base_umbrella.free.iloc[(self.base_umbrella.free['#Coor'] -target).abs().argsort()[:1]]
        near_true = nearest
        x_val = near_true['#Coor']
        y_val = near_true['Prob']
        ax.scatter(x_val, y_val, s=50)
        return None
            
    def plot_probability(self, ax=None, title='Probability Distribution', label=None, errorevery=15):
        if ax is None:
            ax = self.plt_fig()
        if label is None:
            label = self.base_umbrella.system
        indicator = [self.base_umbrella.mean, self.base_umbrella.standard_error]
        df = self.base_umbrella.free
        try:
            ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Prob'],
                     yerr=df.loc[:, '+/-.1'], capsize=2.5, capthick=1.2,
                     linewidth=1.5,errorevery=errorevery, label=f'{label} {indicator[0]:.2f} nm \u00B1 {indicator[1]:.2f} nm')
            if indicator is not None:
                self.prob_plot_indicator(indicator, ax, label=label)
        except:
            ax.plot(df.loc[:, '#Coor'], df.loc[:, 'Prob'], label=label)          
            
    def plot_free_energy_mod(self, negative=False, ax=None, title='Free Energy Profile', c='black', label=None,errorevery=15):
        if ax is None:
            ax = self.plt_fig()
        if label is None:
            label = self.base_umbrella.system
        # if c is None:
        #     c = '#00429d'
        indicator = [self.base_umbrella.mean, self.base_umbrella.standard_error]
        df = self.base_umbrella.free
        try:
            if negative is False:
                ax.errorbar((62 - df.loc[:, '#Coor']), df.loc[:, 'Free'],
                     yerr=df.loc[:, '+/-'], c=c, capsize=2.5, capthick=1.2,
                     linewidth=1.5, errorevery=15)
            else:
                ax.errorbar(-(62 - df.loc[:, '#Coor']), df.loc[:, 'Free'],
                     yerr=df.loc[:, '+/-'], c=c, capsize=2.5, capthick=1.2,
                     linewidth=1.5, errorevery=15)
            # if indicator is not None:
            #     self.plot_indicator(indicator, ax, c=c, label=label)
        except:
            ax.plot(df.loc[:, '#Coor'], df.loc[:, 'Free'], label=label)      
    
             
class CustomObsWham(WhamAnalysis):
    def __init__(self, base_umbrella):
        super().__init__(base_umbrella)
        
    def run_wham(self,opp_umbrella,wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot):
        """
        Run Weighted Histogram Analysis Method on production windows.
        
        Parameters:
            wham_dir (str): Path to wham executable.
            xmin (str): Minimum distance of center of mass order parameter in simulation units.
            xmax (str): Maximum distance of center of mass order parameter in simulation units.
            umbrella_stiff (str): The parameter used to modified the stiffness of the center of mass spring potential
            n_bins (str): number of histogram bins to use.
            tol (str): Convergence tolerance for the WHAM calculations.
            n_boot (str): Number of monte carlo bootstrapping error analysis iterations to preform.

        """
        print('Running WHAM analysis...')
        self.opp_umbrella = opp_umbrella
        
        self.base_umbrella.mod_com_dir = join(self.base_umbrella.production_sim_dir, 'mod_com_dir')
        self.base_umbrella.pos_dir = join(self.base_umbrella.production_sim_dir, 'pos_dir')
        self.base_umbrella.com_dir = join(self.base_umbrella.production_sim_dir, 'com_dir')
        
        copy_com_pos(self.base_umbrella.production_sim_dir, self.base_umbrella.com_dir, self.base_umbrella.pos_dir)
                                          
        self.opp_umbrella.mod_com_dir = join(self.opp_umbrella.production_sim_dir, 'mod_com_dir')
        self.opp_umbrella.pos_dir = join(self.opp_umbrella.production_sim_dir, 'pos_dir')
        self.opp_umbrella.com_dir = join(self.opp_umbrella.production_sim_dir, 'com_dir')        
        
        copy_com_pos(self.opp_umbrella.production_sim_dir, self.opp_umbrella.com_dir, self.opp_umbrella.pos_dir)
             
        value = float(get_xmax(self.base_umbrella.com_dir, self.opp_umbrella.com_dir))
        
        auto_1 = mod_com_info(self.base_umbrella.production_sim_dir, self.base_umbrella.com_dir, self.base_umbrella.pos_dir, self.base_umbrella.mod_com_dir, value)                                   
        auto_2 = mod_com_info(self.opp_umbrella.production_sim_dir, self.opp_umbrella.com_dir, self.opp_umbrella.pos_dir, self.opp_umbrella.mod_com_dir, value)
                                 
        pre_temp = self.base_umbrella.production_sims[0].input.input['T']
        if ('C'.upper() in pre_temp) or ('C'.lower() in pre_temp):
            self.base_umbrella.temperature = (float(pre_temp[:-1]) + 273.15) / 3000
        elif ('K'.upper() in pre_temp) or ('K'.lower() in pre_temp):
             self.base_umbrella.temperature = float(pre_temp[:-1]) / 3000
        two_sided_wham(wham_dir,
                       auto_1,
                       auto_2,
                       self.base_umbrella.mod_com_dir,
                       self.opp_umbrella.mod_com_dir,
                       str(xmin),
                       str(xmax),
                       str(umbrella_stiff),
                       str(n_bins),
                       str(tol),
                       str(n_boot),
                       str(self.base_umbrella.temperature))
                                         
    def to_si(self, n_bins):
        self.base_umbrella.com_dir = join(self.base_umbrella.production_sim_dir, 'mod_com_dir')
        pre_temp = self.base_umbrella.production_sims[0].input.input['T']
        if ('C'.upper() in pre_temp) or ('C'.lower() in pre_temp):
            self.base_umbrella.temperature = (float(pre_temp[:-1]) + 273.15) / 3000
        elif ('K'.upper() in pre_temp) or ('K'.lower() in pre_temp):
             self.base_umbrella.temperature = float(pre_temp[:-1]) / 3000
        free = pd.read_csv(f'{self.base_umbrella.system_dir}/production/mod_com_dir/freefile', sep='\t', nrows=int(n_bins), engine='pyarrow')
        free['Free'] = free['Free'].div(self.base_umbrella.temperature)
        free['+/-'] = free['+/-'].div(self.base_umbrella.temperature)
        free['#Coor'] *= 0.8518
        self.base_umbrella.free = free     

            