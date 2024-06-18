from ipy_oxdna.oxdna_simulation import Simulation, Force, Observable, SimulationManager
from ipy_oxdna.wham_analysis import *
from ipy_oxdna.pymbar_analysis import PymbarAnalysis
from ipy_oxdna.vmmc import VirtualMoveMonteCarlo
import multiprocessing as mp
import os
from os.path import join, exists
import numpy as np
import shutil
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal, norm, t
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import io
import sys
import contextlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed# from numba import jit
import pickle
from json import load, dump
import traceback
import warnings
from scipy.optimize import OptimizeWarning
import pyarrow
import warnings


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
        self.progress = UmbrellaProgress(self)
        self.info_utils = UmbrellaInfoUtils(self)
        self.observables = UmbrellaObservables(self)
        
        self.wham = WhamAnalysis(self)
        self.pymbar = PymbarAnalysis(self)
        
        self.f = Force()
        self.obs = Observable()
        
        self.umbrella_bias = None
        self.com_by_window = None
        self.r0 = None
        
        self.read_progress()
        
    
    def queue_sims(self, simulation_manager, sim_list, continue_run=False):
        for sim in sim_list:
            simulation_manager.queue_sim(sim, continue_run=continue_run)        
     
             
    def wham_run(self, wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot, all_observables=True):
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
                                    
    
    def read_progress(self):
        self.progress.read_pre_equlibration_progress()
        self.progress.read_equlibration_progress()
        self.progress.read_production_progress()
        self.progress.read_wham_progress()
        self.progress.read_convergence_analysis_progress()
        self.progress.read_melting_temperature_progress()


class ComUmbrellaSampling(BaseUmbrellaSampling):
    def __init__(self, file_dir, system, clean_build=False):
        super().__init__(file_dir, system, clean_build=clean_build)

     
    def build_pre_equlibration_runs(self, simulation_manager,  n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters, starting_r0, steps, observable=False, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False, protein=None, force_file=None):
        self.observables_list = []
        self.windows.pre_equlibration_windows(n_windows)
        self.rate_umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows, starting_r0, steps)
        if observable:
            self.initialize_observables(com_list, ref_list, print_every, name, force_energy_split=False)
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
        if observable:
            self.initialize_observables(com_list, ref_list, print_every, name, force_energy_split=False)
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
        if observable:
            self.initialize_observables(com_list, ref_list, print_every, name, force_energy_split=False)
        if continue_run is False:
            self.us_build.build(self.production_sims, input_parameters,
                                self.forces_list, self.observables_list,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file) 
        
        self.queue_sims(simulation_manager, self.production_sims, continue_run=continue_run)         
 

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
        

    def initialize_observables(self, com_list, ref_list, print_every=1e4, name='all_observables.txt', force_energy_split=True):
        self.observables.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        
        if force_energy_split is True:
            number_of_forces = self.info_utils.get_n_external_forces()      
            for idx in range(number_of_forces):
                self.observables.force_energy_observable(print_every=print_every, name=name, print_group=f'force_{idx}')
        else:
            self.observables.force_energy_observable(print_every=print_every, name=name)
                    
        self.observables.kinetic_energy_observable(print_every=print_every, name=name)
        self.observables.potential_energy_observable(print_every=print_every, name=name, split='True')


class MeltingUmbrellaSampling(ComUmbrellaSampling):
    def __init__(self, file_dir, system, clean_build=False):
        super().__init__(file_dir, system, clean_build=clean_build)
        self.hb_by_window = None
        self.potential_energy_by_window = None
    
    
    def build_pre_equlibration_runs(self, simulation_manager,  n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters, starting_r0, steps, observable=False, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False, protein=None, force_file=None, custom_observable=False, force_energy_split=True):
        self.observables_list = []
        self.windows.pre_equlibration_windows(n_windows)
        self.rate_umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows, starting_r0, steps)
        
        if observable:
            self.initialize_observables(com_list, ref_list, print_every, name, force_energy_split=force_energy_split)
        
        if continue_run is False:
            self.us_build.build(self.pre_equlibration_sims, input_parameters,
                                self.forces_list, self.observables_list,cms_observable=custom_observable,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file)
            for sim in self.pre_equlibration_sims:
                sim.build_sim.build_hb_list_file(com_list, ref_list)
        self.queue_sims(simulation_manager, self.pre_equlibration_sims, continue_run=continue_run)
        
        
    def build_equlibration_runs(self, simulation_manager,  n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                                observable=False, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False,
                                protein=None, force_file=None, custom_observable=False, force_energy_split=True):
        self.observables_list = []
        self.windows.equlibration_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)

        if observable:
            self.initialize_observables(com_list, ref_list, print_every, name, force_energy_split=force_energy_split)
        
        if continue_run is False:
            self.us_build.build(self.equlibration_sims, input_parameters,
                                self.forces_list, self.observables_list, cms_observable=custom_observable,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file)
            for sim in self.equlibration_sims:
                sim.build_sim.build_hb_list_file(com_list, ref_list)
        self.queue_sims(simulation_manager, self.equlibration_sims, continue_run=continue_run)
        
        
    def build_production_runs(self, simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                              observable=True, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False,
                              protein=None, force_file=None, custom_observable=False, force_energy_split=True):
        self.observables_list = []
        self.windows.equlibration_windows(n_windows)
        self.windows.production_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)

        if observable:
            self.initialize_observables(com_list, ref_list, print_every, name, force_energy_split=force_energy_split)

        if continue_run is False:
            self.us_build.build(self.production_sims, input_parameters,
                                self.forces_list, self.observables_list, cms_observable=custom_observable,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file)
            for sim in self.production_sims:
                sim.build_sim.build_hb_list_file(com_list, ref_list)
        self.queue_sims(simulation_manager, self.production_sims, continue_run=continue_run)   
        
        
    def initialize_observables(self, com_list, ref_list, print_every=1e4, name='all_observables.txt', force_energy_split=True):
        self.observables.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        self.observables.hb_list_observable(print_every=print_every, only_count='true', name=name)
        
        if force_energy_split is True:
            number_of_forces = self.info_utils.get_n_external_forces()      
            for idx in range(number_of_forces):
                self.observables.force_energy_observable(print_every=print_every, name=name, print_group=f'force_{idx}')
        else:
            self.observables.force_energy_observable(print_every=print_every, name=name)
                    
        self.observables.kinetic_energy_observable(print_every=print_every, name=name)
        self.observables.potential_energy_observable(print_every=print_every, name=name, split='True')
        

    
    def discrete_and_continuous_converg_analysis(self, convergence_slice, temp_range, n_bins, xmin, xmax, umbrella_stiff, max_hb, epsilon=1e-7, reread_files=False, max_iterations=100000):
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
            
  
    def temperature_interpolation(self, xmin, xmax, max_hb, temp_range, reread_files=False, all_observables=False, convergence_slice=None, molcon=None):
    
        if reread_files is False:
            if self.obs_df is None:
                self.analysis.read_all_observables('prod')
        elif reread_files is True:
            self.analysis.read_all_observables('prod')

        number_of_forces = self.info_utils.get_n_external_forces()
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
                        log_e_beta_u[win_idx][temp_idx][hb_idx] = -np.inf
        self.log_e_beta_u = log_e_beta_u
        
        
        # f_i = self.info_utils.get_wham_biases()
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
            self.volume_correction = list(map(self.volume_correction, (box_size,xmax)))
        else:    
            self.volume_correction = np.log((((self.box_size / 2) * np.sqrt(3))**3) / ((4/3) * np.pi * xmax**3))
                
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
    
    def wham_temperature_interpolation(self, temp_range, n_bins, xmin, xmax, umbrella_stiff, max_hb, epsilon=1e-7, reread_files=False, all_observables=False, max_iterations=100000, convergence_slice=None):
        
        if reread_files is False:
            if self.obs_df is None:
                self.analysis.read_all_observables('prod')
        elif reread_files is True:
            self.analysis.read_all_observables('prod')
        
        number_of_forces = self.info_utils.get_n_external_forces()
        force_energy = [f'force_energy_{idx}' for idx in range(number_of_forces)]
        # force_energy = ['force_energy']

        min_length = min([len(inner_list['com_distance']) for inner_list in self.obs_df])
        truncated_com_values = [inner_list['com_distance'][:min_length] for inner_list in self.obs_df] 
        x_range = np.round(np.linspace(xmin, xmax, (len(self.obs_df) + 1), dtype=np.double)[1:], 3)
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

        self.info_utils.get_temperature()
        temperature = np.array(self.temperature, dtype=np.double)
        beta = 1 / temperature


        top_file = self.production_sims[0].sim_files.top
        with open(top_file, 'r') as f:
            n_particles_in_system = np.double(f.readline().split(' ')[0])
        
        n_particles_in_op = max_hb * 2
        
        truncated_non_pot_energy = truncated_kinetic_energy + truncated_force_energy

        new_energy_per_window = self._new_calcualte_bias_energy(truncated_non_pot_energy, temp_range, truncated_potential_energy=truncated_potential_energy)
        new_energy_per_window = np.array(new_energy_per_window) * n_particles_in_system

        new_energy_per_window_reshaped = np.array(new_energy_per_window).swapaxes(0,1)
        
        calculated_bin_centers, bin_edges = self.get_bins(xmin, xmax, n_bins=n_bins)

        self.info_utils.get_r0_values()       #Calculate the biases in the windows
        window_biases = np.array([[
            self.w_i(bin_value, r0_value, umbrella_stiff)
            for bin_value in calculated_bin_centers
            ] 
            for r0_value in self.r0
        ], dtype=np.double)

        #Get the com values
        all_com_values = np.array(truncated_com_values)

        
        # Initialize the 4D list
        weights_in_bins = [[[[] for _ in range(len(bin_edges) -1 )] for _ in range(len(self.obs_df)) ] for _ in range(len(temp_range))] 

        bin_idx = np.digitize(all_com_values, bin_edges) - 1  # -1 because np.digitize starts from 1
        # return bin_idx
        # Populate the weights in bins
        for temp_idx, temp_bias in enumerate(new_energy_per_window_reshaped):
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
                        log_histogram[temp_idx, window_idx, bin_idx] = -np.inf

        self.log_histogram = log_histogram

        norm_factor = np.max(logsumexp(log_histogram, axis=2), axis=1)
        
        p_i_b_s = np.exp(log_histogram - norm_factor[:, np.newaxis, np.newaxis])
        
        summed_p_i_b_s = np.sum(p_i_b_s, axis=2)
        summed_log_histogram = logsumexp(log_histogram, axis=2)
        
        
        beta_range_reshaped = beta_range[:, np.newaxis, np.newaxis]
        

        #The numerator of p_x is the sum of the counts from each window
        numerator = np.sum(p_i_b_s, axis=1)
        log_numerator = logsumexp(log_histogram, axis=1)
        
        rng = np.random.default_rng()    
        # epsilon = 1e-7

        f_i_bias_factor = np.array([np.exp(-window_biases * bet) for bet in beta_range])
        log_f_i_bias_factor = np.array([(-window_biases * bet) for bet in beta_range])
        
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
                exponential_term = intermediate_result * beta_range[:, np.newaxis, np.newaxis]
                log_denominator = logsumexp(summed_log_histogram[:, :, np.newaxis] + exponential_term, axis=1)
                                
                #Compute the probability of each bin
                log_p_x = log_numerator - log_denominator

                #Recompute the f_i values per window. This value will update till convergence
                log_sum_p_bf = logsumexp(log_p_x[:, np.newaxis, :] + log_f_i_bias_factor, axis=2)

                f_i_temps_new = -temp_range_scaled[:,np.newaxis] * log_sum_p_bf

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
        p_x = np.exp(log_p_x)
        free = -log_p_x
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
    
    
    def _weight_sample_pre_compute_input(self, truncated_com_distance, truncated_potential_energy, truncated_force_energy, beta, n_particles_in_system, n_restraints):
        
        def force_potential_energy(r0, com_position):
            return 0.5 * 5 * (com_position - r0)**2
        
        r0_list = self.r0
        truncated_com_distance = np.concatenate(truncated_com_distance)
        
        com_grid, r0_grid = np.meshgrid(truncated_com_distance, r0_list)
        
        #I have all hypotheical window energies in the shape total_data x n_windows
        all_window_energies = force_potential_energy(r0_grid, com_grid) * 2 / n_particles_in_system
        all_window_energies =  all_window_energies.swapaxes(1,0)
        
        #Pot energies in the shape total_data x n_windows
        all_pot_energies = np.concatenate(np.sum(truncated_potential_energy, axis=2))
        
        #Harmonic trap energies in the shape total_data x n_windows
        all_force_energies = truncated_force_energy[:,:,:n_restraints]
        all_force_energies = np.sum(all_force_energies, axis=2)
        all_force_energies = np.concatenate(all_force_energies)
        
        #All window and energies and harmonic trap energies
        all_biasing_energies = all_window_energies + all_force_energies[:, np.newaxis]
        
        #All energies in the shape total_data x n_windows
        all_energies = all_pot_energies[:, np.newaxis] - all_biasing_energies
        
        log_biasing_function = all_energies * beta
        biasing_function = np.exp(all_energies * beta)
        
        return biasing_function, log_biasing_function, all_window_energies, all_pot_energies, all_force_energies

    
    def weight_sample(self, epsilon=1e-15, reread_files=False, convergence_slice=None):
        
        if reread_files is False:
            if self.obs_df is None:
                self.analysis.read_all_observables('prod')
        elif reread_files is True:
            self.analysis.read_all_observables('prod')
             
        number_of_forces = self.info_utils.get_n_external_forces()
        n_restraints = number_of_forces - 2
        
        force_energy = [f'force_energy_{idx}' for idx in range(number_of_forces)]

        names = ['backbone', 'bonded_excluded_volume', 'stacking', 'nonbonded_excluded_volume', 'hydrogen_bonding', 'cross_stacking', 'coaxial_stacking', 'debye_huckel']

        min_length = min([len(inner_list['com_distance']) for inner_list in self.obs_df])        
        truncated_force_energy = [inner_list[force_energy][:min_length] for inner_list in self.obs_df]
        truncated_potential_energy = [inner_list[names][:min_length] for inner_list in self.obs_df]
        truncated_com_distance = [inner_list['com_distance'][:min_length] for inner_list in self.obs_df]
        
        if convergence_slice is not None:
            truncated_force_energy = [inner_list[convergence_slice] for inner_list in truncated_force_energy]

        temperature = np.array(self.temperature, dtype=np.double)
        beta = 1 / temperature
        
        top_file = self.production_sims[0].sim_files.top
        with open(top_file, 'r') as f:
            n_particles_in_system = np.double(f.readline().split(' ')[0])
        
        truncated_force_energy = np.array(truncated_force_energy)
        truncated_potential_energy = np.array(truncated_potential_energy)
        truncated_com_distance = np.array(truncated_com_distance)
        
        self.info_utils.get_r0_values()
        
        num_samples_per_window = truncated_force_energy.shape[1]
        n_windows = truncated_force_energy.shape[0]

        biasing_function, log_biasing_function, all_window_energies, all_pot_energies, all_force_energies = self._weight_sample_pre_compute_input(truncated_com_distance, truncated_potential_energy, truncated_force_energy, beta, n_particles_in_system, n_restraints)
        return all_window_energies, all_pot_energies, all_force_energies
        constant_term = biasing_function * num_samples_per_window
        
        # f_i_old = np.ones(biasing_function.shape[1])
        # rho_old = np.ones([n_windows, int(biasing_function.shape[0]/n_windows)])

        log_f_i_old = np.ones(biasing_function.shape[1])

        convergence_criterion = 1
        iteration = 0
        update_frequency = 5
        significant_digits = abs(int(np.floor(np.log10(abs(epsilon))))) +1
        custom_bar_format = '{desc} {r_bar}'
        first = True

        with tqdm(desc='Non-parametric WHAM', leave=True, bar_format=custom_bar_format) as pbar:
            while convergence_criterion > epsilon:
                
                log_rho_new = np.log(1) - logsumexp(log_biasing_function + log_f_i_old, b=num_samples_per_window, axis=1)
                log_fi_new = np.log(1) - logsumexp(log_biasing_function + log_rho_new[:, np.newaxis], axis=0)
                
                # rho_new = 1 / np.sum(constant_term * f_i_old, axis=1)
                # f_i_new = 1 / np.sum(biasing_function * rho_new[:, np.newaxis], axis=0)
                
                # f_i_new, rho_new = single_iteration(f_i_old, biasing_function, constant_term, num_samples_per_window)
                convergence_criterion = np.max(np.abs(log_fi_new - log_f_i_old))

                log_fi_old = log_fi_new
                # f_i_old = f_i_new
                iteration +=1

                if (iteration % update_frequency == 0) or (first == True):
                    formatted_convergence = f"{convergence_criterion:.{significant_digits}f} / {float(epsilon)}"
                    pbar.set_postfix_str(f"Convergence: {formatted_convergence}")
                    if first is True:
                        pbar.update(0)
                    else:
                        pbar.update(update_frequency)
                first = False
            
        # f_i_new = f_i_new - np.min(f_i_new)
        # rho_new = 1 / np.sum(constant_term * f_i_new, axis=1)
        # rho_new = rho_new / np.sum(rho_new)
         
        return log_rho_new, log_fi_new


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
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            self.popt, _ = curve_fit(self.sigmoid, temp_range, self.inverted_finfs, p0, method='dogbox')
    
        # Generate fitted data
        self.x_fit = np.linspace(min(temp_range), max(temp_range), 500)
        self.y_fit = self.sigmoid(self.x_fit, *self.popt)
        
        
        idx = np.argmin(np.abs(self.y_fit - 0.5))
        self.Tm = self.x_fit[idx]

    
    def volume_correction(self, box_size, xmax):
        return np.log((((box_size / 2) * np.sqrt(3))**3) / ((4/3) * np.pi * xmax**3))
    

    def entropic_correction(self, bin_values):
        return self.temperature * np.log(4 * np.pi * np.power(bin_values, 2))


    # Function to convert molar concentration to box size
    def molar_concentration_to_box_size(self, molcon):
        box = ((2/(molcon*6.0221415*10**23))**(1/3)) / 8.5179*10.0**(9)
        return box
    
    
    def box_size_to_molar_concentration(self, box_size):
        molcon = (2/(8.5179*10.0**(-9)*box_size)**3)/(6.0221415*10**23)
        return molcon
    
    
    def mg_concentration_to_na_concentration(self, mg_concentration):
        na_concentration = 120 * mg_concentration**(0.5)
        return na_concentration
    
    
    def na_tm_to_mg_tm(self, na_tm, mg_concentration, fGC, Nbp):
        """_summary_
        citation: Owczarzy, Richard, et al. "Predicting stability of DNA duplexes in solutions containing magnesium and monovalent cations." Biochemistry 47.19 (2008): 5336-5353.
        Args:
            na_tm (_type_): _description_
            mg_concentration (_type_): _description_
            fGC (_type_): _description_
            Nbp (_type_): _description_

        Returns:
            _type_: _description_
        """
        log_mg = np.log(mg_concentration)
        a = 3.92e-5
        b = -9.11e-6
        c = 6.26e-5
        d = 1.42e-5
        e = -4.82e-4
        f = 5.25e-4
        g = 8.31e-5
        
        t1 = 1 / (na_tm + 273.15)
        t2 = a
        t3 = b * log_mg
        t4 = fGC * (c + d * log_mg)
        t5_1 = 1 / (2 * (Nbp - 1))
        t5_2 = e + f * log_mg + g * (log_mg)**2
        
        # print(f'{t1=}')
        # print(f'{t2=}')
        # print(f'{t3=}')
        # print(f'{t4=}')
        # print(f'{t5_1=}')
        # print(f'{t5_2=}')
        
        one_over_mg_tm = t1 + t2 + t3 + t4 + t5_1 * t5_2
        mg_tm = (1 / one_over_mg_tm) - 273.15
        
        return mg_tm
        
        
    def sigmoid(self, x, L, x0, k, b):
        return L / (1 + np.exp(-k * (x - x0))) + b
    
    
    def w_i(self, r, r0, stiff):
        return 0.5 * float(stiff) * (r - r0)**2


    def celcius_to_scaled(self, temp):
        return (temp + 273.15) / 3000

            
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
            if hasattr(sim.sim_files, 'hb_contacts') is False:
                continue
            df = pd.read_csv(sim.sim_files.hb_contacts, header=None, engine='pyarrow')
            self.hb_contacts_by_window[idx] = df


class UmbrellaBuild:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
    
    def build(self, sims, input_parameters, forces_list, observables_list,
              observable=False, sequence_dependant=False, cms_observable=False, protein=None, force_file=None):

        self.prepare_simulation_environment(sims)
        
        # Using ThreadPoolExecutor to parallelize the simulation building
        warnings.filterwarnings(
        "ignore",
        "os.fork\\(\\) was called\\. os\\.fork\\(\\) is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock\\.",
        RuntimeWarning
        )
        workers = self.base_umbrella.info_utils.get_number_of_processes() - 1
        with ProcessPoolExecutor(max_workers=workers) as executor:
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
            if (protein is not None) and (protein is not False):
                sim.add_protein_par()
            if (force_file is not None) and  (force_file is not False):
                sim.add_force_file()
            for force in forces:
                sim.add_force(force)
            if observable:
                for observables in observables_list:
                    sim.add_observable(observables)
            if (cms_observable is not False) and (cms_observable is not None):
                for cms_obs_dict in cms_observable:
                    sim.oxpy_run.cms_obs(cms_obs_dict['idx'],
                                         name=cms_obs_dict['name'],
                                         print_every=cms_obs_dict['print_every'])
            if sequence_dependant:
                sim.make_sequence_dependant()
            sim.sim_files.parse_current_files()
        except Exception as e:
            error_traceback = traceback.format_exc() 
            print(error_traceback) # Gets the full traceback
            raise e

    def parallel_force_group_name(self, sims):
        workers = self.base_umbrella.info_utils.get_number_of_processes() - 1
        with ProcessPoolExecutor(max_workers=workers) as executor:
            executor.map(self.process_simulation, sims)

    def process_simulation(self, sim):
        sim.sim_files.parse_current_files()
        with open(sim.sim_files.force, 'r') as f:
            force_js = load(f)
        force_js_modified = {key: {'group_name': key, **value} for key, value in force_js.items()}
        with open(sim.sim_files.force, 'w') as f:
            dump(force_js_modified, f, indent=4)       


    def prepare_simulation_environment(self, sims):
        umbrella_stage = sims[0].sim_dir.split('/')[-2]
        
        if exists(join(self.base_umbrella.system_dir, umbrella_stage)) and bool(os.listdir(join(self.base_umbrella.system_dir, umbrella_stage))):
            if self.base_umbrella.clean_build is True:
                answer = input('Are you sure you want to delete all simulation files? Type y/yes to continue or anything else to return use UmbrellaSampling(clean_build=str(force) to skip this message')
                if answer.lower() not in ['y', 'yes']:
                    sys.exit('\nContinue a previous umbrella simulation using:\ncontinue_run=int(n_steps)')
            
                elif answer.lower() in ['y', 'yes']:
                    shutil.rmtree(join(self.base_umbrella.system_dir, umbrella_stage))
                    os.mkdir(join(self.base_umbrella.system_dir, umbrella_stage))  
            
            elif self.base_umbrella.clean_build == False:
                sys.exit('\nThe simulation directory already exists, if you wish to write over the directory set:\nUmbrellaSampling(clean_build=str(force)).\n\nTo continue a previous umbrella simulation use:\ncontinue_run=int(n_steps)')
                
                
            elif self.base_umbrella.clean_build in 'force':
                shutil.rmtree(join(self.base_umbrella.system_dir, umbrella_stage))
                os.mkdir(join(self.base_umbrella.system_dir, umbrella_stage))

        return None
    
    
    def modify_topology_for_unique_pairing(self):
        """
        Modify the topology file to ensure that each nucleotide can only bind to its original partner.
        """
        for sim in self.base_umbrella.equlibration_sims:
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


class UmbrellaInfoUtils:
    def __init__(self, base_umbrella):
        self.base = base_umbrella
        
    def get_wham_biases(self):
        #I need to get the freefile
        #The free file is in the com_dim
        with open(f'{self.base.com_dir}/freefile', 'r') as f:
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
        for idx,sim in enumerate(self.base.production_sims):
            sim.sim_files.parse_current_files()
            df = pd.read_csv(sim.sim_files.com_distance, header=None, engine='pyarrow', dtype=np.double, names=['com_distance'])
            com_distance_by_window[idx] = df
        self.base.com_by_window = com_distance_by_window
                
    def get_r0_values(self):
        self.base.r0 = []
        force_files = [sim.sim_files.force for sim in self.base.equlibration_sims]
        for force_file in force_files:
            with open(force_file, 'r') as f:
                force_js = load(f)
            forces = list(force_js.keys())
            self.base.r0.append(float(force_js[forces[-1]]['r0']))
            
    def get_stiff_value(self):
        force_files = [sim.sim_files.force for sim in self.base.equlibration_sims]
        for force_file in force_files:
            with open(force_file, 'r') as f:
                force_js = load(f)
            forces = list(force_js.keys())
            break
        self.base.stiff = float(force_js[forces[-1]]['stiff'])
        
    def get_temperature(self):
        pre_temp = self.base.equlibration_sims[0].input.input['T']
        if ('C'.upper() in pre_temp) or ('C'.lower() in pre_temp):
            self.base.temperature = (float(pre_temp[:-1]) + 273.15) / 3000
        elif ('K'.upper() in pre_temp) or ('K'.lower() in pre_temp):
             self.base.temperature = float(pre_temp[:-1]) / 3000
             
    def get_n_particles_in_system(self):
        top_file = self.base.equlibration_sims[0].sim_files.top
        with open(top_file, 'r') as f:
            self.base.n_particles_in_system = np.double(f.readline().split(' ')[0])
             
    def get_n_windows(self):
        
        try:
            self.base.n_windows = len(self.base.production_sims)
        except:
            pass
        try:
            self.base.n_windows = len(self.base.equlibration_sims)
        except:
            pass
        try:
            self.base.n_windows = len(self.base.pre_equlibration_sims)
        except:
            print('No simulations found')
            
            
    def get_n_external_forces(self):
        try:
            force_file = [file for file in os.listdir(self.base.file_dir) if file.endswith('force.txt')][0]
            force_file = f'{self.base.file_dir}/{force_file}'
        
            number_of_forces = 0

            with open(force_file, 'r') as f:
                for line in f:
                    if '{' in line:
                        number_of_forces += 1
        except:
            number_of_forces = 0
            
        return number_of_forces + 2
    
    
    def get_number_of_processes(self):
        try:
            # Try using os.sched_getaffinity() available on some Unix systems
            if sys.platform.startswith('linux'):
                return len(os.sched_getaffinity(0))
            else:
                # Fallback to multiprocessing.cpu_count() which works cross-platform
                return mp.cpu_count()
        except Exception as e:
            # Handle possible exceptions (e.g., no access to CPU info)
            print(f"Failed to determine the number of CPUs: {e}")
            return 1  # Safe fallback if number of CPUs can't be determined
            
            
    def copy_last_conf_from_eq_to_prod(self):
        for eq_sim, prod_sim in zip(self.base.equlibration_sims, self.base.production_sims):
            shutil.copyfile(eq_sim.sim_files.last_conf, f'{prod_sim.sim_dir}/last_conf.dat')


class UmbrellaProgress:
    def __init__(self, base_umbrella):
        self.base = base_umbrella
        
        
    def read_pre_equlibration_progress(self):
        if exists(join(self.base.system_dir, 'pre_equlibration')):
            self.base.pre_equlibration_sim_dir = join(self.base.system_dir, 'pre_equlibration')
            n_windows = len(os.listdir(self.base.pre_equlibration_sim_dir))
            self.base.pre_equlibration_sims = []
            for window in range(n_windows):
                self.base.pre_equlibration_sims.append(Simulation(join(self.base.pre_equlibration_sim_dir, str(window)), join(self.base.pre_equlibration_sim_dir, str(window))))

            
    def read_equlibration_progress(self):
        if exists(join(self.base.system_dir, 'equlibration')):
            self.base.equlibration_sim_dir = join(self.base.system_dir, 'equlibration')
            self.base.n_windows = len(os.listdir(self.base.equlibration_sim_dir))
            self.base.equlibration_sims = []
            for window in range(self.base.n_windows):
                self.base.equlibration_sims.append(Simulation(join(self.base.equlibration_sim_dir, str(window)), join(self.base.equlibration_sim_dir, str(window))))
    
    
    def read_production_progress(self):              
        if exists(join(self.base.system_dir, 'production')):
            self.base.production_sim_dir = join(self.base.system_dir, 'production')
            n_windows = len(self.base.equlibration_sims)
            self.base.production_window_dirs = [join(self.base.production_sim_dir, str(window)) for window in range(n_windows)]
            
            self.base.production_sims = []
            for s, window_dir, window in zip(self.base.equlibration_sims, self.base.production_window_dirs, range(n_windows)):
                self.base.production_sims.append(Simulation(self.base.equlibration_sims[window].sim_dir, str(window_dir))) 
    
    
    def read_wham_progress(self):          
        if exists(join(self.base.system_dir, 'production', 'com_dir', 'freefile')):
            self.base.com_dir = join(self.base.system_dir, 'production', 'com_dir')
            with open(join(self.base.production_sim_dir, 'com_dir', 'freefile'), 'r') as f:
                file = f.readlines()
            file = [line for line in file if not line.startswith('#')]
            self.base.n_bins = len(file)
            
            with open(join(self.base.com_dir, 'metadata'), 'r') as f:
                lines = [line.split(' ') for line in f.readlines()]
            
            self.base.wham.xmin = float(lines[0][1])
            self.base.wham.xmax = float(lines[-1][1])
            self.base.wham.umbrella_stiff = float(lines[0][-1])
            self.base.wham.n_bins = self.base.n_bins
            # self.wham.get_n_data_per_com_file()
            self.base.free = self.base.wham.to_si(self.base.n_bins, self.base.com_dir)
            self.base.mean = self.base.wham.w_mean(self.base.free)
            try:
                self.base.standard_error, self.base.confidence_interval = self.base.wham.bootstrap_w_mean_error(self.base.free)
            except:
                self.base.standard_error, self.base.confidence_interval = ('failed', 'failed')
        
        
    def read_convergence_analysis_progress(self):   
        if exists(join(self.base.system_dir, 'production', 'com_dir', 'convergence_dir')):
            try:
                self.base.convergence_dir = join(self.base.com_dir, 'convergence_dir')
                self.base.chunk_convergence_analysis_dir = join(self.base.convergence_dir, 'chunk_convergence_analysis_dir')
                self.base.data_truncated_convergence_analysis_dir = join(self.base.convergence_dir, 'data_truncated_convergence_analysis_dir')  
                self.base.wham.chunk_dirs = [join(self.base.chunk_convergence_analysis_dir, chunk_dir) for chunk_dir in os.listdir(self.base.chunk_convergence_analysis_dir)]
                self.base.wham.data_truncated_dirs = [join(self.data_truncated_convergence_analysis_dir, chunk_dir) for chunk_dir in os.listdir(self.base.data_truncated_convergence_analysis_dir)]
                
                self.base.chunk_dirs_free = [self.base.wham.to_si(self.base.n_bins, chunk_dir) for chunk_dir in self.base.wham.chunk_dirs]
                self.base.chunk_dirs_mean = [self.base.wham.w_mean(free_energy) for free_energy in self.base.chunk_dirs_free]
                try:
                    self.base.chunk_dirs_standard_error, self.base.chunk_dirs_confidence_interval = zip(
                        *[self.base.wham.bootstrap_w_mean_error(free_energy) for free_energy in self.base.chunk_dirs_free]
                    )
                except:
                    self.base.chunk_dirs_standard_error = ['failed' for _ in range(len(self.base.chunk_dirs_free))]
                    self.base.chunk_dirs_confidence_interval = ['failed' for _ in range(len(self.base.chunk_dirs_free))]
    
                    
                self.base.data_truncated_free = [self.base.wham.to_si(self.base.n_bins, chunk_dir) for chunk_dir in self.base.wham.data_truncated_dirs]
                self.base.data_truncated_mean = [self.base.wham.w_mean(free_energy) for free_energy in self.base.data_truncated_free]
                try:
                    self.base.data_truncated_standard_error, self.data_truncated_confidence_interval = zip(
                        *[self.base.wham.bootstrap_w_mean_error(free_energy) for free_energy in self.data_truncated_free]
                    )
                except:
                    self.base.data_truncated_standard_error = ['failed' for _ in range(len(self.base.data_truncated_free))]
                    self.base.data_truncated_confidence_interval = ['failed' for _ in range(len(self.base.data_truncated_free))]
            except:
                pass
    
    
    def read_melting_temperature_progress(self):
        if exists(join(self.base.system_dir, 'production', 'vmmc_dir')):
            self.base.vmmc_dir = join(self.base.system_dir, 'production', 'vmmc_dir')
            self.base.vmmc_sim = VirtualMoveMonteCarlo(self.base.file_dir, self.base.vmmc_dir)
            self.base.vmmc_sim.analysis.read_vmmc_op_data()
            self.base.vmmc_sim.analysis.calculate_sampling_and_probabilities()
            self.base.vmmc_sim.analysis.calculate_and_estimate_melting_profiles()
            

class UmbrellaObservables:
    def __init__(self, base_umbrella):
        self.base = base_umbrella
        self.obs = Observable()
        
        
    def com_distance_observable(self, com_list, ref_list,  print_every=1e4, name='com_distance.txt'):
        """ Build center of mass observable"""
        com_observable = self.obs.distance(
            particle_1=com_list,
            particle_2=ref_list,
            print_every=f'{print_every}',
            name=f'{name}',
            PBC='1'
        )  
        self.base.observables_list.append(com_observable)
        
        
    def hb_list_observable(self, print_every=1e4, name='hb_observable.txt', only_count='true'):
        hb_obs = self.obs.hb_list(
            print_every=str(print_every),
            name=name,
            only_count='true'
           )
        self.base.observables_list.append(hb_obs)
        
        
    def force_energy_observable(self, print_every=1e4, name='force_energy.txt', print_group=None, precision=12, general_format='true'):
        force_energy_obs = self.obs.force_energy(
            print_every=str(print_every),
            name=str(name),
            print_group=print_group,
            precision=precision,
            general_format=general_format
        )
        self.base.observables_list.append(force_energy_obs)
    
    
    def kinetic_energy_observable(self, print_every=1e4, name='kinetic_energy.txt'):
        kin_obs = self.obs.kinetic_energy(
            print_every=str(print_every),
            name=str(name)
        )
        self.base.observables_list.append(kin_obs)
        
        
    def potential_energy_observable(self, print_every=1e4, name='potential_energy.txt', split='True', precision=12, general_format='true'):
        """ Build potential energy observable for temperature interpolation"""
        pot_obs = self.obs.potential_energy(
            print_every=str(print_every),
            split=str(split),
            name=str(name),
            precision=precision,
            general_format=general_format
        )
        self.base.observables_list.append(pot_obs)


    
class UmbrellaAnalysis:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
        self.base_umbrella.obs_df = None
    
    
    def plot_OPs(self, op_strings, rolling_window=1, n_bins=50, dpi=300):
        fig, ax = plt.subplots(2, len(op_strings), figsize=(15, 6), dpi=dpi)
        for idx, df in enumerate(self.base_umbrella.obs_df):
            for string_idx, op_string in enumerate(op_strings):
                try:
                    ax[0,string_idx].plot(df['steps'], df[op_string].rolling(rolling_window).mean())
                except:
                    pass

                try:
                    H, bins = np.histogram(df[op_string], bins=n_bins, density=True)
                    H = H * (bins[1] - bins[0])
                    ax[1,string_idx].plot(bins[:-1], H)
                except:
                    pass

                if idx == len(self.base_umbrella.obs_df) - 1:
                    ax[0,string_idx].set_ylabel(op_string)
                    ax[0,string_idx].set_xlabel('Steps')
                    ax[1,string_idx].set_ylabel('Probability')
                    ax[1,string_idx].set_xlabel(op_string)
                    
        fig.tight_layout()

    
    def plot_melting_CVs(self, rolling_window=1, n_bins=50, dpi=300):
        try:
            fig, ax = plt.subplots(2, 4, figsize=(15, 6), dpi=dpi)

            for idx, df in enumerate(self.base_umbrella.obs_df):
                ax[0,0].plot(df['steps'], df['com_distance'].rolling(rolling_window).mean())
                H, bins = np.histogram(df['com_distance'], bins=n_bins, density=True)
                H = H * (bins[1] - bins[0])
                ax[1,0].plot(bins[:-1], H)

                ax[0,1].plot(df['steps'], df['hb_list'].rolling(rolling_window).mean())
                H, bins = np.histogram(df['hb_list'], bins=n_bins, density=True)
                H = H * (bins[1] - bins[0])

                ax[1,1].plot(bins[:-1], H)
                try:
                    ax[0,2].plot(df['steps'], df['hb_contact'].rolling(rolling_window).mean())
                    H, bins = np.histogram(df['hb_contact'], bins=n_bins, density=True)
                    H = H * (bins[1] - bins[0])
                    ax[1,2].plot(bins[:-1], H)
                except:
                    pass

                ax[0,3].plot(df['steps'], df.filter(like='force_energy').sum(axis=1).rolling(rolling_window).mean())
                H, bins = np.histogram(df.filter(like='force_energy').sum(axis=1), bins=n_bins, density=True)
                H = H * (bins[1] - bins[0])
                ax[1,3].plot(bins[:-1], H)


            ax[0,0].set_ylabel('Center of Mass Distance')
            ax[0,1].set_ylabel('Number of H-bonds')
            ax[0,3].set_ylabel('Force Energy')
            ax[0,2].set_ylabel('HB Contacts / Max HBs')

            ax[0,0].set_xlabel('Steps')
            ax[0,1].set_xlabel('Steps')
            ax[0,3].set_xlabel('Steps')
            ax[0,2].set_xlabel('Steps')

            ax[1,0].set_ylabel('Probability')
            ax[1,1].set_ylabel('Probability')
            ax[1,3].set_ylabel('Probability')
            ax[1,2].set_ylabel('Probability')

            ax[1,0].set_xlabel('Center of Mass Distance')
            ax[1,1].set_xlabel('Number of H-bonds')
            ax[1,3].set_xlabel('Force Energy')
            ax[1,2].set_xlabel('HB Contacts / Max HBs')

        except Exception as e:
            plt.clf()
            fig, ax = plt.subplots(2, 3, figsize=(10, 6), dpi=dpi)

            for idx, df in enumerate(self.base_umbrella.obs_df):
                ax[0,0].plot(df['steps'], df['com_distance'].rolling(rolling_window).mean())
                H, bins = np.histogram(df['com_distance'], bins=n_bins, density=True)
                ax[1,0].plot(bins[:-1], H)

                ax[0,1].plot(df['steps'], df['hb_list'].rolling(rolling_window).mean())
                H, bins = np.histogram(df['hb_list'], bins=n_bins, density=True)
                ax[1,1].plot(bins[:-1], H)

                ax[0,2].plot(df['steps'], df.filter(like='force_energy').sum(axis=1).rolling(rolling_window).mean())
                H, bins = np.histogram(df.filter(like='force_energy').sum(axis=1), bins=n_bins, density=True)
                ax[1,2].plot(bins[:-1], H)

            ax[0,0].set_ylabel('Center of Mass Distance')
            ax[0,1].set_ylabel('Number of H-bonds')
            ax[0,2].set_ylabel('Force Energy')

            ax[0,0].set_xlabel('Steps')
            ax[0,1].set_xlabel('Steps')
            ax[0,2].set_xlabel('Steps')

            ax[1,0].set_ylabel('Count')
            ax[1,1].set_ylabel('Count')
            ax[1,2].set_ylabel('Count')

            ax[1,0].set_xlabel('Center of Mass Distance')
            ax[1,1].set_xlabel('Number of H-bonds')
            ax[1,2].set_xlabel('Force Energy')

        fig.tight_layout()
         

    def read_all_observables(self, sim_type):
        file_name = self.base_umbrella.observables_list[0]['output']['name']
        print_every = int(float(self.base_umbrella.observables_list[0]['output']['print_every']))
        
        # Determine the simulation list based on sim_type
        if sim_type == 'eq':
            sim_list = self.base_umbrella.equlibration_sims
        elif sim_type == 'prod':
            sim_list = self.base_umbrella.production_sims
        elif sim_type == 'pre_eq':
            sim_list = self.base_umbrella.pre_equlibration_sims

        # Assuming force_js is accessible and relevant here
        # with open(sim_list[0].sim_files.force, 'r') as f:
        #     force_js = load(f)
        #     number_of_forces = len(force_js.keys())
        
        number_of_forces = 0
        with open(sim_list[0].sim_files.observables, 'r') as f:
            obs_file = load(f)
        obs_file_cols = obs_file['output']['cols']
        for col in obs_file_cols:
            if 'force_energy' in col['type']:
                number_of_forces += 1
        
        names = ['backbone', 'bonded_excluded_volume', 'stacking', 'nonbonded_excluded_volume', 'hydrogen_bonding', 'cross_stacking', 'coaxial_stacking', 'debye_huckel']

        force_energy = [f'force_energy_{idx}' for idx in range(number_of_forces)]
        
        class_name = type(self.base_umbrella).__name__
        if class_name == 'MeltingUmbrellaSampling':
            columns = ['com_distance', 'hb_list', *force_energy, 'kinetic_energy', *names]
        else:
            columns = ['com_distance', *force_energy, 'kinetic_energy', *names]
            
        # Parallel processing using ProcessPoolExecutor
        warnings.filterwarnings(
        "ignore",
        "os.fork\\(\\) was called\\. os\\.fork\\(\\) is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock\\.",
        RuntimeWarning
        )
        workers = self.base_umbrella.info_utils.get_number_of_processes()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            try:
                results = list(tqdm(executor.map(self.process_simulation, sim_list, 
                                            [file_name]*len(sim_list), 
                                            [number_of_forces]*len(sim_list), 
                                            [columns]*len(sim_list), 
                                            [print_every]*len(sim_list)), total=len(sim_list), desc='Reading Observables'))
            except Exception as e:
                print(f"An error occurred: {e}")

        self.base_umbrella.obs_df = results    
            
        return self.base_umbrella.obs_df

    def process_simulation(self, sim, file_name, number_of_forces, columns, print_every):
        """
        Standalone function to process a single simulation.
        """
        try:
            observable = pd.read_csv(f"{sim.sim_dir}/{file_name}", header=None, engine='pyarrow')
        except (FileNotFoundError, pyarrow.lib.ArrowInvalid) as e:
            print(f'No Data found in all_observables.txt file at:\n {sim.sim_dir}')
            observable = pd.DataFrame()
        if not observable.empty:
            data = [list(filter(lambda a: a != '', row[0].split(' '))) for row in observable.itertuples(index=False)]
            df = pd.DataFrame(data, columns=columns, dtype=np.double)
            df['steps'] = np.arange(len(df)) * print_every
        else:
            df = pd.DataFrame(columns=columns + ['steps'])
        if hasattr(sim.sim_files, 'hb_contacts'):
            try:
                sim.sim_files.parse_current_files()
                hb_contact = pd.read_csv(sim.sim_files.hb_contacts, header=None, engine='pyarrow')
                df['hb_contact'] = hb_contact
            except Exception as e:
                pass
            
        return df
        
    def view_observable(self, sim_type, idx, sliding_window=False, observable=None):
        if observable == None:
            observable=self.base_umbrella.observables_list[0]
        
        if sim_type == 'pre_eq':
            self.base_umbrella.pre_equlibration_sims[idx].analysis.plot_observable(observable, fig=False, sliding_window=sliding_window)
        
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
        for sim in self.base_umbrella.production_sims[idxes]:    
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
    
    def view_last_conf(self, sim_type, window):
        if sim_type == 'pre_eq':
            try:
                self.base_umbrella.pre_equlibration_sims[window].analysis.view_last()
            except:
                self.base_umbrella.pre_equlibration_sims[window].analysis.view_init()
                
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
        self.base_umbrella.info_utils.get_temperature()
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
    

    def convergence_analysis(self, n_chunks, data_added_per_iteration, wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot):
        """
        Split your data into a set number of chunks to check convergence.
        If all chunks are the same your free energy profile if probabily converged
        Also create datasets with iteraterativly more data, to check convergence progress
        """
        self.wham_dir = wham_dir
        self.xmin = xmin
        self.xmax = xmax
        self.umbrella_stiff = umbrella_stiff
        self.n_bins = n_bins
        self.tol = tol
        self.n_boot = n_boot
        
        if not exists(self.base_umbrella.com_dir):
            self.base_umbrella.wham_run(wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot)
            
        self.base_umbrella.convergence_dir = join(self.base_umbrella.com_dir, 'convergence_dir')
        self.base_umbrella.chunk_convergence_analysis_dir = join(self.base_umbrella.convergence_dir, 'chunk_convergence_analysis_dir')
        self.base_umbrella.data_truncated_convergence_analysis_dir = join(self.base_umbrella.convergence_dir, 'data_truncated_convergence_analysis_dir')
        
        if exists(self.base_umbrella.convergence_dir):
            shutil.rmtree(self.base_umbrella.convergence_dir)
            
        if not exists(self.base_umbrella.convergence_dir):
            os.mkdir(self.base_umbrella.convergence_dir)
            os.mkdir(self.base_umbrella.chunk_convergence_analysis_dir)
            os.mkdir(self.base_umbrella.data_truncated_convergence_analysis_dir) 
  
        self.chunk_convergence_analysis(n_chunks)
        
        self.base_umbrella.chunk_dirs_free = [self.to_si(self.n_bins, chunk_dir) for chunk_dir in self.chunk_dirs]
        self.base_umbrella.chunk_dirs_mean = [self.w_mean(free_energy) for free_energy in self.base_umbrella.chunk_dirs_free]
        try:
            self.base_umbrella.chunk_dirs_standard_error, self.base_umbrella.chunk_dirs_confidence_interval = zip(
                *[self.bootstrap_w_mean_error(free_energy) for free_energy in self.base_umbrella.chunk_dirs_free]
            )
        except:
            self.base_umbrella.chunk_dirs_standard_error = ['failed' for _ in range(len(self.base_umbrella.chunk_dirs_free))]
            self.base_umbrella.chunk_dirs_confidence_interval = ['failed' for _ in range(len(self.base_umbrella.chunk_dirs_free))]


        self.data_truncated_convergence_analysis(data_added_per_iteration)
        
        self.base_umbrella.data_truncated_free = [self.to_si(self.n_bins, chunk_dir) for chunk_dir in self.data_truncated_dirs]
        self.base_umbrella.data_truncated_mean = [self.w_mean(free_energy) for free_energy in self.base_umbrella.data_truncated_free]
        try:
            self.base_umbrella.data_truncated_standard_error, self.base_umbrella.data_truncated_confidence_interval = zip(
                *[self.bootstrap_w_mean_error(free_energy) for free_energy in self.base_umbrella.data_truncated_free]
            )
        except:
            self.base_umbrella.data_truncated_standard_error = ['failed' for _ in range(len(self.base_umbrella.data_truncated_free))]
            self.base_umbrella.data_truncated_confidence_interval = ['failed' for _ in range(len(self.base_umbrella.data_truncated_free))]
        
        return None
    
    
    def to_si(self, n_bins, com_dir):
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
            
            
@contextlib.contextmanager
def suppress_output():
    new_stdout = io.StringIO()
    new_stderr = io.StringIO()
    old_stderr = sys.stderr
    old_stdout = sys.stdout
    try:
        sys.stderr = new_stderr
        sys.stdout = new_stdout
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
with suppress_output():
    import pymbar
