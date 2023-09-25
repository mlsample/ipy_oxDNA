from oxdna_simulation import Simulation, Force, Observable, SimulationManager
from wham_analysis import *
import multiprocessing as mp
import os
from os.path import join, exists
import numpy as np
import shutil
import pandas as pd
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
import scienceplots



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
    
    def queue_sims(self, simulation_manager, sim_list, continue_run=False):
        for sim in sim_list:
            simulation_manager.queue_sim(sim, continue_run=continue_run)        
            
    def wham_run(self, wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot):
<<<<<<< HEAD
<<<<<<< Updated upstream
=======
=======
>>>>>>> 7a099b60142c38b617b557731d8f34de4512dfa0
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
<<<<<<< HEAD
        self.wham.wham_dir = wham_dir
        self.wham.xmin = xmin
        self.wham.xmax = xmax
        self.wham.umbrella_stiff = umbrella_stiff
        self.wham.n_bins = n_bins
        self.wham.tol = tol
        self.wham.n_boot = n_boot
        
>>>>>>> Stashed changes
=======
>>>>>>> 7a099b60142c38b617b557731d8f34de4512dfa0
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
        if exists(join(self.system_dir, 'equlibration')):
            self.equlibration_sim_dir = join(self.system_dir, 'equlibration')
            n_windows = len(os.listdir(self.equlibration_sim_dir))
            self.equlibration_sims = []
            for window in range(n_windows):
                self.equlibration_sims.append(Simulation(join(self.equlibration_sim_dir, str(window)), join(self.equlibration_sim_dir, str(window))))
                
        if exists(join(self.system_dir, 'production')):
            self.production_sim_dir = join(self.system_dir, 'production')
            self.production_window_dirs = [join(self.production_sim_dir, str(window)) for window in range(n_windows)]
            n_windows = len(self.equlibration_sims)
            self.production_sims = []
            for s, window_dir, window in zip(self.equlibration_sims, self.production_window_dirs, range(n_windows)):
                self.production_sims.append(Simulation(self.equlibration_sims[window].sim_dir, str(window_dir))) 
        
        if exists(join(self.system_dir, 'production', 'com_dir', 'freefile')):
            self.com_dir = join(self.system_dir, 'production', 'com_dir')
            with open(join(self.production_sim_dir, 'com_dir', 'freefile'), 'r') as f:
                file = f.readlines()
            file = [line for line in file if not line.startswith('#')]
            self.n_bins = len(file)
            self.wham.get_n_data_per_com_file()
            self.free = self.wham.to_si(self.n_bins, self.com_dir)
            self.mean = self.wham.w_mean(self.free)
            try:
                self.standard_error, self.confidence_interval = self.wham.bootstrap_w_mean_error(self.free)
            except:
                self.standard_error, self.confidence_interval = ('failed', 'failed')
        
        if exists(join(self.system_dir, 'production', 'com_dir', 'convergence_dir')):
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
        data = [line[1] for line in data]
 
        return data
   
    def get_com_distance_by_window(self):
        com_distance_by_window = {}
        for idx,sim in enumerate(self.production_sims):
            sim.sim_files.parse_current_files()
            df = pd.read_csv(sim.sim_files.com_distance, header=None)
            com_distance_by_window[idx] = df
        self.com_by_window = com_distance_by_window

    
    def get_bias_potential_value(self, xmin, xmax, n_windows, stiff):
        x_range = np.round(np.linspace(xmin, xmax, (n_windows + 1))[1:], 3)
        umbrella_bias = [0.5 * stiff * (com_values - eq_pos)**2 for com_values, eq_pos in zip(self.com_by_window.values(), x_range)]
        self.umbrella_bias = umbrella_bias
    
    def copy_last_conf_from_eq_to_prod(self):
        for eq_sim, prod_sim in zip(self.equlibration_sims, self.production_sims):
            shutil.copyfile(eq_sim.sim_files.last_conf, f'{prod_sim.sim_dir}/last_conf.dat')

            
class UmbrellaBuild:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
    
    def build(self, sims, input_parameters, forces_list, observables_list,
              observable=False, sequence_dependant=False, cms_observable=False, protein=None, force_file=None):
        
        if exists(self.base_umbrella.system_dir):
            if self.base_umbrella.clean_build is True:
                answer = input('Are you sure you want to delete all simulation files? Type y/yes to continue or anything else to return use UmbrellaSampling(clean_build=str(force) to skip this message')
                if (answer == 'y') or (answer == 'yes'):
                    pass
                else:
                    sys.exit('\nRemove optional argument clean_build and continue a previous umbrella simulation using:\nsimulation_manager.run(continue_run=int(n_steps))')    
            elif self.base_umbrella.clean_build == 'force':                    
                    pass
            elif self.base_umbrella.clean_build == False:
                sys.exit('\nThe simulation directory already exists, if you wish to write over the directory set:\nUmbrellaSampling(clean_build=str(force)).\n\nTo continue a previous umbrella simulation use:\nsimulation_manager.run(continue_run=int(n_steps))')  
            
        for sim, forces in zip(sims, forces_list):
            sim.build(clean_build='force')
            
            if protein is not None:
                sim.add_protein_par()
            if force_file is not None:
                sim.add_force_file()
            for force in forces:
                sim.add_force(force)
            if observable == True:
                for observables in observables_list:
                    sim.add_observable(observables)
            if cms_observable is not False:
                for cms_obs_dict in cms_observable:
                    sim.oxpy_run.cms_obs(cms_obs_dict['idx'],
                                         name=cms_obs_dict['name'],
                                         print_every=cms_obs_dict['print_every'])
            sim.input_file(input_parameters)
            if sequence_dependant is True:
                sim.sequence_dependant()
                

class ComUmbrellaSampling(BaseUmbrellaSampling):
    def __init__(self, file_dir, system, clean_build=False):
        super().__init__(file_dir, system, clean_build=clean_build)
        self.observables_list = []
        
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
        self.windows.equlibration_windows(n_windows)
        self.windows.production_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        self.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        if continue_run is False:
            self.us_build.build(self.production_sims, input_parameters,
                                self.forces_list, self.observables_list,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file) 
            
            # self.validate_production_build(n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
            #            observable, sequence_dependant, print_every, name, continue_run,
            #            protein, force_file)
        
        self.queue_sims(simulation_manager, self.production_sims, continue_run=continue_run)   
    
    
#     def validate_production_build(self, simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
#                        observable, sequence_dependant, print_every, name, continue_run,
#                        protein, force_file):
#         #I want to check if all of the required files are in the window folder for the equlibration and production, other wise return a print stament leting me know that the build has failed.
#         for sim in self.production_sims:
#             sim.sim_files.parse_current_files()
        
#         #First, I can get all of the files in each production sim folder ??
#         if force_file is not None:
#             for sim in self.production_sims:
#                 sim.sim_file
            
        
    
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
        
    def build_equlibration_runs(self, simulation_manager,  n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                                observable=False, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False,
                                protein=None, force_file=None):
        self.windows.equlibration_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        self.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        self.hb_list_observable(print_every=print_every, only_count='true')
        if continue_run is False:
            self.us_build.build(self.equlibration_sims, input_parameters,
                                self.forces_list, self.observables_list,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file)
            for sim in self.equlibration_sims:
                sim.build_sim.build_hb_list_file(com_list, ref_list)
        self.queue_sims(simulation_manager, self.equlibration_sims, continue_run=continue_run)
        
        
    def build_production_runs(self, simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                              observable=True, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False,
                              protein=None, force_file=None):
        self.windows.equlibration_windows(n_windows)
        self.windows.production_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        self.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        self.hb_list_observable(print_every=print_every, only_count='true')
        if continue_run is False:
            self.us_build.build(self.production_sims, input_parameters,
                                self.forces_list, self.observables_list,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file)
            for sim in self.production_sims:
                sim.build_sim.build_hb_list_file(com_list, ref_list)
        self.queue_sims(simulation_manager, self.production_sims, continue_run=continue_run)   
    
    def hb_list_observable(self, print_every=1e4, name='hb_observable.txt', only_count='true'):
        """ Build center of mass observable"""
        hb_obs = self.obs.hb_list(
            print_every='1e3',
            name='hb_observable.txt',
            only_count='true'
           )
        self.observables_list.append(hb_obs)
        
    def copy_hb_list_to_com_dir(self):
        copy_h_bond_files(self.production_sim_dir, self.com_dir)

    def get_hb_list_by_window(self):
        hb_list_by_window = {}
        for idx,sim in enumerate(self.production_sims):
            sim.sim_files.parse_current_files()
            df = pd.read_csv(sim.sim_files.hb_observable, header=None)
            hb_list_by_window[idx] = df
        self.hb_by_window = hb_list_by_window
            
    
    def unbias_com_to_hb(self, xmin, xmax, n_windows, stiff, max_hb):
        if self.com_by_window is None:
            self.get_com_distance_by_window()
        if self.hb_by_window is None:
            self.get_hb_by_window()
        if self.umbrella_bias is None:
            self.get_bias_potential_value(xmin, xmax, n_windows, stiff)

        
        com_by_window = self.com_by_window
        hb_by_window = self.hb_by_window
        umbrella_bias = self.umbrella_bias
        
        unbiased_discrete_window = {idx:np.zeros(int(max_hb + 1)) for idx in range(n_windows)}
        for idx in range(n_windows):
            index_to_add_at = np.array(hb_by_window[idx].values.T[0])
            
            biases = np.array([value if value != 0 else 1 for value in umbrella_bias[idx].values.T[0]])
            value_to_add = 1 / biases
            
            np.add.at(unbiased_discrete_window[idx], index_to_add_at, value_to_add)
        
        print(len(com_by_window[0]))
        print(len(hb_by_window[0]))
        print(len(umbrella_bias[0]))
        print(unbiased_discrete_window)
        self.unbiased_discrete_windows = unbiased_discrete_window

    def make_last_hist_files(self):
        for idx,sim in enumerate(self.production_sims):
            hist = self.unbiased_discrete_windows[idx]
            with open(join(sim.sim_dir, 'last_hist.dat'), 'w') as f:
                f.write(f'#t = 0 {sim.input.input["T"]} \n')
                for idx, n_hb in enumerate(hist):
                    f.write(f"{idx} {n_hb} {n_hb} \n")
    
    def run_wham_discete(self, max_hb):
        invocation = 'python3 '
        script_location = '/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples/duplex_melting/us_melting_52_no_non_canonical/wham.py 2 '
        wfile_location = '/scratch/mlsample/ipy_oxDNA/ipy_oxdna_examples/duplex_melting/us_melting_52_no_non_canonical/wfile.txt '
        last_hist_location = [join(sim.sim_dir, 'last_hist.dat') for sim in self.production_sims]
        
        invocation += script_location
        for last_hist_path in last_hist_location:
            invocation += wfile_location
            invocation += (last_hist_path + ' ')
        x = subprocess.check_output(invocation, shell=True)
        print(x.decode())
        # hbs = map(float, x.decode().split()[-(max_hb + 1)*2:][::2])
        # prob = map(float, x.decode().split()[-(max_hb + 1)*2:][1::2])
        # self.discrete_hist = {key:value for key,value in zip(hbs, prob)}
        
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



class UmbrellaAnalysis:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
        
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
            df = pd.read_csv(f"{self.sim.sim_dir}/{file_name}", header=None)
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
    
    def equlibration_windows(self, n_windows):
        """ 
        Sets a attribute called equlibration_sims containing simulation objects for all equlibration windows.
        
        Parameters:
            n_windows (int): Number of umbrella sampling windows.
        """
        self.base_umbrella.equlibration_sim_dir = join(self.base_umbrella.system_dir, 'equlibration')
        if not exists(self.base_umbrella.equlibration_sim_dir):
            os.mkdir(self.base_umbrella.equlibration_sim_dir)
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
    
    
    def spawn(self, f, args=()):
        """Spawn subprocess"""
        p = mp.Process(target=f, args=args)
        p.start()
        if self.join == True:
            p.join()
        self.process = p
    
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
        free = pd.read_csv(f'{self.base_umbrella.system_dir}/production/mod_com_dir/freefile', sep='\t', nrows=int(n_bins))
        free['Free'] = free['Free'].div(self.base_umbrella.temperature)
        free['+/-'] = free['+/-'].div(self.base_umbrella.temperature)
        free['#Coor'] *= 0.8518
        self.base_umbrella.free = free     

            