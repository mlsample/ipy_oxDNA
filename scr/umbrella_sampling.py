from oxdna_simulation import Simulation, Force, Observable, SimulationManager
from wham_analysis import wham_analysis
import multiprocessing as mp
import os
from os.path import join, exists
import numpy as np
import shutil
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class BaseUmbrellaSampling:
    def __init__(self, file_dir, system):
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
    
    def queue_sims(self, simulation_manager, sim_list, continue_run=False):
        for sim in sim_list:
            simulation_manager.queue_sim(sim, continue_run=continue_run)        
            
    def wham_job(self, wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot):
        self.wham.run_wham(wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot)
        self.wham.to_si(n_bins)
        self.wham.w_mean()
        self.wham.bootstrap_w_mean_error()
    
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
            with open(join(self.production_sim_dir, 'com_dir', 'freefile'), 'r') as f:
                file = f.readlines()
            file = [line for line in file if not line.startswith('#')]
            n_bins = len(file)
            self.wham.to_si(n_bins)
            self.wham.w_mean()
            self.wham.bootstrap_w_mean_error()
            

class ComUmbrellaSampling(BaseUmbrellaSampling):
    def __init__(self, file_dir, system):
        super().__init__(file_dir, system)
        
        
    def build_equlibration_runs(self, simulation_manager,  n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                                observable=False, print_every=1e4, name='com_distance.txt', continue_run=False):
        self.windows.equlibration_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        self.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        if continue_run is False:
            self.us_build.build(self.equlibration_sims, input_parameters, self.forces_list, self.observables_list, observable=observable)
        self.queue_sims(simulation_manager, self.equlibration_sims, continue_run=continue_run)
        
        
    def build_production_runs(self, simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                              observable=True, print_every=1e4, name='com_distance.txt', continue_run=False):
        self.windows.equlibration_windows(n_windows)
        self.windows.production_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        self.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)
        if continue_run is False:
            self.us_build.build(self.production_sims, input_parameters, self.forces_list, self.observables_list, observable=observable)
        self.queue_sims(simulation_manager, self.production_sims, continue_run=continue_run)   
    
    
    def com_distance_observable(self, com_list, ref_list,  print_every=1e4, name='com_distance.txt'):
        """ Build center of mass observable"""
        self.observables_list = []
        obs = Observable()
        com_observable = obs.distance(
            particle_1=com_list,
            particle_2=ref_list,
            print_every=f'{print_every}',
            name=f'{name}'
        )  
        self.observables_list.append(com_observable)
 

    def umbrella_forces(self, com_list, ref_list, stiff, xmin, xmax, n_windows):
        """ Build Umbrella potentials"""
        x_range = np.linspace(xmin, xmax, (n_windows + 1))[1:]
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
    
    
    def plot_free(self):
        self.wham.plot_free_energy()


class UmbrellaAnalysis:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
        
    def view_observable(self, sim_type, idx, sliding_window=False):
        if sim_type == 'eq':
            self.base_umbrella.equlibration_sims[idx].analysis.plot_observable(self.base_umbrella.observables_list[0], fig=False, sliding_window=sliding_window)

        if sim_type == 'prod':
            self.base_umbrella.production_sims[idx].analysis.plot_observable(self.base_umbrella.observables_list[0], fig=False, sliding_window=sliding_window)
    
    def hist_observable(self, sim_type, idx, bins=10):
        if sim_type == 'eq':
            self.base_umbrella.equlibration_sims[idx].analysis.hist_observable(self.base_umbrella.observables_list[0],
                                                                               fig=False, bins=bins)

        if sim_type == 'prod':
            self.base_umbrella.production_sims[idx].analysis.hist_observable(self.base_umbrella.observables_list[0],
                                                                             fig=False, bins=bins)
    
    def view_observables(self, sim_type, sliding_window=False):
        if sim_type == 'eq':
            plt.figure()
            for sim in self.base_umbrella.equlibration_sims:
                sim.analysis.plot_observable(self.base_umbrella.observables_list[0], fig=False, sliding_window=sliding_window)

        if sim_type == 'prod':
            plt.figure(figsize=(15,3))
            for sim in self.base_umbrella.production_sims:
                sim.analysis.plot_observable(self.base_umbrella.observables_list[0], fig=False, sliding_window=sliding_window)
    
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
    
    
class UmbrellaBuild:
    def __init__(self, base_umbrella):
        pass
    
    def build(self, sims, input_parameters, forces_list, observables_list, observable=False):
        for sim, forces in zip(sims, forces_list):
            sim.build(clean_build='force')
            for force in forces:
                sim.add_force(force)
            if observable == True:
                for observables in observables_list:
                    sim.add_observable(observables)
            sim.input_file(input_parameters)


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
    
    
    def to_si(self, n_bins):
        self.base_umbrella.com_dir = join(self.base_umbrella.production_sim_dir, 'com_dir')
        pre_temp = self.base_umbrella.production_sims[0].input.input['T']
        if ('C'.upper() in pre_temp) or ('C'.lower() in pre_temp):
            self.base_umbrella.temperature = (float(pre_temp[:-1]) + 273.15) / 3000
        elif ('K'.upper() in pre_temp) or ('K'.lower() in pre_temp):
             self.base_umbrella.temperature = float(pre_temp[:-1]) / 3000
        free = pd.read_csv(f'{self.base_umbrella.system_dir}/production/com_dir/freefile', sep='\t', nrows=int(n_bins))
        free['Free'] = free['Free'].div(self.base_umbrella.temperature)
        free['+/-'] = free['+/-'].div(self.base_umbrella.temperature)
        free['#Coor'] *= 0.8518
        self.base_umbrella.free = free     
    
    
    def w_mean(self):
        free = self.base_umbrella.free.loc[:, 'Free']
        coord = self.base_umbrella.free.loc[:, '#Coor']
        prob = np.exp(-free) / sum(np.exp(-free))
        mean = sum(coord * prob)
        self.base_umbrella.mean = mean
    
    
    def bootstrap_w_mean_error(self):
        coord = self.base_umbrella.free.loc[:, '#Coor']
        free = self.base_umbrella.free.loc[:, 'Free'] 
        prob = np.exp(-free) / sum(np.exp(-free))
    
        err = self.base_umbrella.free.loc[:, '+/-']
        mask = np.isnan(err)
        err[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), err[~mask])
        cov = np.diag(err**2)
    
        estimate = np.array(multivariate_normal.rvs(mean=free, cov=cov, size=10000, random_state=None))
        est_prob = [np.exp(-est) / sum(np.exp(-est)) for est in estimate]
        means = [sum(coord * e_prob) for e_prob in est_prob]
        standard_error = np.std(means)
        self.base_umbrella.standard_error = standard_error
        
    def plt_fig(self, title='Free Energy Profile', xlabel='End-to-End Distance (nm)', ylabel='Free Energy / k$_B$T'):
        from matplotlib.ticker import MultipleLocator
        plt.figure(dpi=200, figsize=(5.5, 4.5))
        plt.title(title)
        plt.xlabel(xlabel, size=12)
        plt.ylabel(ylabel, size=12)
        #plt.rcParams['text.usetex'] = True
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['xtick.minor.size'] = 4
        plt.rcParams['ytick.major.size'] = 6
        #plt.rcParams['ytick.minor.size'] = 4
        plt.rcParams['axes.linewidth'] = 1.25
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        ax = plt.gca()
        ax.set_aspect('auto')
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        #ax.yaxis.set_minor_locator(MultipleLocator(2.5))
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.tick_params(axis='both', which='minor', labelsize=9)
        ax.yaxis.set_ticks_position('both')
        return ax

    
    def plot_indicator(self, indicator, ax, c=None, label=None):
        target = indicator[0]
        nearest = self.base_umbrella.free.iloc[(self.base_umbrella.free['#Coor'] -target).abs().argsort()[:1]]
        near_true = nearest
        x_val = near_true['#Coor']
        y_val = near_true['Free']
        ax.scatter(x_val, y_val, s=50, c=c, label=f'{label} {target:.2f} nm \u00B1 {indicator[1]:.2f} nm')
        return None
    
    
    def plot_free_energy(self, ax=None, title='Free Energy Profile', c=None, label=None):
        if ax is None:
            ax = self.plt_fig()
        if label is None:
            label = self.base_umbrella.system
        # if c is None:
        #     c = '#00429d'
        indicator = [self.base_umbrella.mean, self.base_umbrella.standard_error]
        df = self.base_umbrella.free
        ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Free'],
                     yerr=df.loc[:, '+/-'], c=c, capsize=2.5, capthick=1.2,
                     linewidth=1.5, errorevery=15)
        if indicator is not None:
            self.plot_indicator(indicator, ax, c=c, label=label)
        plt.legend()      
    
       
                

            
            
            
            
            
            
            
            
            
            
# class BaseUmbrellaSampling:
#     """
#     Run multiprocess multi-GPU oxDNA umbrella sampling in two steps. 
#         1. Build and run equlibration simulation of umbrella sampling windows.
#         2. Build and run production simulations from the last conf of the equlibration simulations.
#     A initial relaxed top and dat file in file_dir are required.
#     """
#     def __init__(self, file_dir, system):
#         """
#         Initalize the directory containing the inital oxDNA top and dat files, and name the system.
    
#         Parameters:
#             file_dir (str): Path to the directory containing the oxDNA top and dat files.
#             system (str): Name of the umbrella sampling system.
#         """
#         self.system = system
#         self.file_dir = file_dir
#         self.system_dir = join(self.file_dir, self.system)
#         if not exists(self.system_dir):
#             os.mkdir(self.system_dir)  
#         self.read_progress()
    
    
#     def build_and_queue_sims(self, sims, input_parameters, f_1s, f_2s, multi_sys_manager=None, continue_run=False, observable=False):
#         if not multi_sys_manager:
#             manager = SimulationManager()
#         else:
#             manager = multi_sys_manager
#         for sim, f_1, f_2 in zip(sims, f_1s, f_2s):
#             if continue_run == False:
#                 sim.build(clean_build='force')
#                 sim.add_force(f_1)
#                 sim.add_force(f_2)
#             if observable == True:
#                 sim.add_observable(self.com_observable)
#             sim.input_file(input_parameters)
#             manager.queue_sim(sim, continue_run=continue_run)
#         if not multi_sys_manager:
#             manager.worker_manager()       
    
#     def equlibration_windows(self, n_windows):
#         """ 
#         Sets a attribute called equlibration_sims containing simulation objects for all equlibration windows.
        
#         Parameters:
#             n_windows (int): Number of umbrella sampling windows.
#         """
#         self.equlibration_sim_dir = join(self.system_dir, 'equlibration')
#         if not exists(self.equlibration_sim_dir):
#             os.mkdir(self.equlibration_sim_dir)
#         self.equlibration_sims = [Simulation(self.file_dir,join(self.equlibration_sim_dir, str(window))) for window in range(n_windows)]
    

#     def equlibration_runs(self, com_list, ref_list, stiff, xmin, xmax, n_windows, input_parameters, multi_sys_manager=False, continue_run=False, observable=False):
#         """
#         Build and run equlibration window simulations
        
#         First, the attribute containing a list of simulation objects called self.equlibration_sims is set by equlibration_windows.
#         Then, an attributes containing a dictonaries with the parameters containing the umbrella potentials are set.
#         Next the com_observable is initalized.
#         Finally the equlibration simulations are build and queued in a simulation manager.
        
#         Parameters:
#             com_list (str): Comma seperated string of oxDNA nucleotide indexes constituting one half of the order parameter.
#             ref_list (str): Comma seperated string of oxDNA nucleotide indexes constituting one half of the order parameter.
#             stiff (str): The parameter used to modified the stiffness of the center of mass spring potential
#             xmin (str): Minimum distance of center of mass order parameter in simulation units.
#             xmax (str): Maximum distance of center of mass order parameter in simulation units.
#             w_windows (str): number of umbrella windows.
#             input_parameter (dict): Used to set the number of steps and other non-default oxDNA input file parameters used.
#             multi_sys_manager (SimulationManager): A provided simulation manager used to run multiple umbrella systems in parallel.
#             continue_run (bool): If true, continue the simulations starting from the last conf
#             observable (bool): If false, do not use an observable for equlibration simulations (speeds up runs)
#         """
#         self.equlibration_windows(n_windows)
#         self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
#         self.com_distance_observable(com_list, ref_list)
        
#         self.build_and_queue_sims(self.equlibration_sims,
#                                   input_parameters,
#                                   self.umbrella_forces_1,
#                                   self.umbrella_forces_2,
#                                   multi_sys_manager=multi_sys_manager,
#                                   continue_run=continue_run,
#                                   observable=observable)
            
#     def production_windows(self, n_windows):
#         """ 
#         Sets a attribute called production_sims containing simulation objects for all production windows.
        
#         Parameters:
#             n_windows (int): Number of umbrella sampling windows.
#         """
#         self.production_sim_dir = join(self.system_dir, 'production')
#         if not exists(self.production_sim_dir):
#             os.mkdir(self.production_sim_dir)
#         self.production_window_dirs = [join(self.production_sim_dir, str(window)) for window in range(n_windows)]
#         self.production_sims = []
#         for s, window_dir, window in zip(self.equlibration_sims, self.production_window_dirs, range(n_windows)):
#             self.production_sims.append(Simulation(self.equlibration_sims[window].sim_dir, str(window_dir)))
   
#     def production_runs(self, com_list, ref_list, stiff, xmin, xmax, n_windows, input_parameters, multi_sys_manager=False, continue_run=False, observable=True):
#         """
#         Build and run produciton window simulations
        
#         Parameters:
#             com_list (str): Comma seperated string of oxDNA nucleotide indexes constituting one half of the order parameter.
#             ref_list (str): Comma seperated string of oxDNA nucleotide indexes constituting one half of the order parameter.
#             stiff (str): The parameter used to modified the stiffness of the center of mass spring potential
#             xmin (str): Minimum distance of center of mass order parameter in simulation units.
#             xmax (str): Maximum distance of center of mass order parameter in simulation units.
#             w_windows (str): number of umbrella windows.
#             input_parameter (dict): Used to set the number of steps and other non-default oxDNA input file parameters used.
#             multi_sys_manager (SimulationManager): A provided simulation manager used to run multiple umbrella systems in parallel.
#             continue_run (bool): If true, continue the simulations starting from the last conf
#             observable (bool): If false, do not use an observable for equlibration simulations (speeds up runs)
#         """
#         self.equlibration_windows(n_windows)
#         self.production_windows(n_windows)
#         self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
#         self.com_distance_observable(com_list, ref_list)
        
#         self.build_and_queue_sims(self.production_sims,
#                                   input_parameters,
#                                   self.umbrella_forces_1,
#                                   self.umbrella_forces_2,
#                                   multi_sys_manager=multi_sys_manager,
#                                   continue_run=continue_run,
#                                   observable=observable) 
        
#     def run_wham(self, wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot):
#         """
#         Run Weighted Histogram Analysis Method on production windows.
        
#         Parameters:
#             wham_dir (str): Path to wham executable.
#             xmin (str): Minimum distance of center of mass order parameter in simulation units.
#             xmax (str): Maximum distance of center of mass order parameter in simulation units.
#             umbrella_stiff (str): The parameter used to modified the stiffness of the center of mass spring potential
#             n_bins (str): number of histogram bins to use.
#             tol (str): Convergence tolerance for the WHAM calculations.
#             n_boot (str): Number of monte carlo bootstrapping error analysis iterations to preform.

#         """
#         self.com_dir = join(self.production_sim_dir, 'com_dir')
#         pre_temp = self.production_sims[0].input.input['T']
#         if ('C'.upper() in pre_temp) or ('C'.lower() in pre_temp):
#             self.temperature = (float(pre_temp[:-1]) + 273.15) / 3000
#         elif ('K'.upper() in pre_temp) or ('K'.lower() in pre_temp):
#              self.temperature = float(pre_temp[:-1]) / 3000
#         wham_analysis(wham_dir,
#                       self.production_sim_dir,
#                       self.com_dir,
#                       str(xmin),
#                       str(xmax),
#                       str(umbrella_stiff),
#                       str(n_bins),
#                       str(tol),
#                       str(n_boot),
#                       str(self.temperature))
     
#     def umbrella_forces(self, com_list, ref_list, stiff, xmin, xmax, n_windows):
#         """ Build Umbrella potentials"""
#         f = Force()
#         x_range = np.linspace(xmin, xmax, (n_windows + 1))[1:]

#         self.umbrella_forces_1 = []
#         self.umbrella_forces_2 = []
        
#         for x_val in x_range:   
#             self.umbrella_force_1 = f.com_force(
#                 com_list=com_list,                        
#                 ref_list=ref_list,                        
#                 stiff=f'{stiff}',                    
#                 r0=f'{x_val}',                       
#                 PBC='1',                         
#                 rate='0',
#             )        
#             self.umbrella_forces_1.append(self.umbrella_force_1)
            
#             self.umbrella_force_2= f.com_force(
#                 com_list=ref_list,                        
#                 ref_list=com_list,                        
#                 stiff=f'{stiff}',                    
#                 r0=f'{x_val}',                       
#                 PBC='1',                          
#                 rate='0',
#             )
#             self.umbrella_forces_2.append(self.umbrella_force_2)
            
    # def com_distance_observable(self, com_list, ref_list,  print_every=1e4, name='com_distance.txt'):
    #     """ Build center of mass observable"""
    #     obs = Observable()
    #     self.com_observable = obs.distance(
    #         particle_1=com_list,
    #         particle_2=ref_list,
    #         print_every=f'{print_every}',
    #         name=f'{name}'
    #     )  
              
#     def finsh_uncomplete_equlibration_run(self, com_list, ref_list, stiff, xmin, xmax, n_windows, input_parameters, multi_sys_manager=False, continue_run=False, observable=False):
#         self.equlibration_windows(n_windows)
#         self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
#         self.com_distance_observable(com_list, ref_list)
#         no_traj_file = []
#         for sim in self.equlibration_sims:
#             if not hasattr(sim.sim_files, 'traj'):
#                 no_traj_file.append(sim)
#         if not no_traj_file:
#             print('All windows have trajectory files')
#             return None
        
#         self.build_and_queue_sims(self.production_sims,
#                                   input_parameters,
#                                   self.umbrella_forces_1,
#                                   self.umbrella_forces_2,
#                                   multi_sys_manager=multi_sys_manager,
#                                   continue_run=continue_run,
#                                   observable=observable) 
            
#     def finsh_uncomplete_prouction_run(self, com_list, ref_list, stiff, xmin, xmax, n_windows, input_parameters, multi_sys_manager=False, continue_run=False, observable=True):
#         self.equlibration_windows(n_windows)
#         self.production_windows(n_windows)
#         self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
#         self.com_distance_observable(com_list, ref_list)
#         no_traj_file = []
#         for sim in self.production_sims:
#             if not hasattr(sim.sim_files, 'traj'):
#                 no_traj_file.append(sim)
#         if not no_traj_file:
#             print('All windows have trajectory files')
#             return None
        
#         self.build_and_queue_sims(self.production_sims,
#                                   input_parameters,
#                                   self.umbrella_forces_1,
#                                   self.umbrella_forces_2,
#                                   multi_sys_manager=multi_sys_manager,
#                                   continue_run=continue_run,
#                                   observable=observable)        
    
#     def read_progress(self):   
#         if exists(join(self.system_dir, 'equlibration')):
#             self.equlibration_sim_dir = join(self.system_dir, 'equlibration')
#             n_windows = len(os.listdir(self.equlibration_sim_dir))
#             self.equlibration_sims = []
#             for window in range(n_windows):
#                 self.equlibration_sims.append(Simulation(join(self.equlibration_sim_dir, str(window)), join(self.equlibration_sim_dir, str(window))))
                
#         if exists(join(self.system_dir, 'production')):
#             self.production_sim_dir = join(self.system_dir, 'production')
#             self.production_window_dirs = [join(self.production_sim_dir, str(window)) for window in range(n_windows)]
#             n_windows = len(self.equlibration_sims)
#             self.production_sims = []
#             for s, window_dir, window in zip(self.equlibration_sims, self.production_window_dirs, range(n_windows)):
#                 self.production_sims.append(Simulation(self.equlibration_sims[window].sim_dir, str(window_dir)))                       
    
#     def to_si(self, n_bins):
#         pre_temp = self.production_sims[0].input.input['T']
#         if ('C'.upper() in pre_temp) or ('C'.lower() in pre_temp):
#             self.temperature = (float(pre_temp[:-1]) + 273.15) / 3000
#         elif ('K'.upper() in pre_temp) or ('K'.lower() in pre_temp):
#              self.temperature = float(pre_temp[:-1]) / 3000
#         free = pd.read_csv(f'{self.system_dir}/production/com_dir/time_series/freefile', sep='\t', nrows=int(n_bins))
#         free['Free'] = free['Free'].div(self.temperature)
#         free['+/-'] = free['+/-'].div(self.temperature)
#         free['#Coor'] *= 0.8518
#         self.free = free     
    
#     def w_mean(self):
#         free = self.free.loc[:, 'Free']
#         coord = self.free.loc[:, '#Coor']
#         prob = np.exp(-free) / sum(np.exp(-free))
#         mean = sum(coord * prob)
#         self.mean = mean
    
    
#     def bootstrap_w_mean_error(self):
#         coord = self.free.loc[:, '#Coor']
#         free = self.free.loc[:, 'Free'] 
#         prob = np.exp(-free) / sum(np.exp(-free))
    
#         err = self.free.loc[:, '+/-']
#         mask = np.isnan(err)
#         err[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), err[~mask])
#         cov = np.diag(err**2)
    
#         estimate = np.array(multivariate_normal.rvs(mean=free, cov=cov, size=10000, random_state=None))
#         est_prob = [np.exp(-est) / sum(np.exp(-est)) for est in estimate]
#         means = [sum(coord * e_prob) for e_prob in est_prob]
#         standard_error = np.std(means)
#         self.standard_error = standard_error
        
#     def plt_fig(self, title=None):
#         from matplotlib.ticker import MultipleLocator
#         plt.figure(dpi=200, figsize=(5.5, 4.5))
#         plt.title(title)
#         plt.xlabel('End-to-End Distance (nm)', size=12)
#         plt.ylabel('Free Energy / k$_B$T', size=12)
#         #plt.rcParams['text.usetex'] = True
#         plt.rcParams['xtick.direction'] = 'in'
#         plt.rcParams['ytick.direction'] = 'in'
#         plt.rcParams['xtick.major.size'] = 6
#         plt.rcParams['xtick.minor.size'] = 4
#         plt.rcParams['ytick.major.size'] = 6
#         #plt.rcParams['ytick.minor.size'] = 4
#         plt.rcParams['axes.linewidth'] = 1.25
#         plt.rcParams['mathtext.fontset'] = 'stix'
#         plt.rcParams['font.family'] = 'STIXGeneral'
#         ax = plt.gca()
#         ax.set_aspect('auto')
#         ax.xaxis.set_minor_locator(MultipleLocator(5))
#         #ax.yaxis.set_minor_locator(MultipleLocator(2.5))
#         ax.tick_params(axis='both', which='major', labelsize=9)
#         ax.tick_params(axis='both', which='minor', labelsize=9)
#         ax.yaxis.set_ticks_position('both')
#         return ax
    
#     def plt_err(system, ax, fmt='-', c=None, label=None):
#         df = free_energy[system]
#         ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Free'],
#                      yerr=df.loc[:, '+/-'], c=c, capsize=2.5, capthick=1.2, fmt=fmt,
#                      linewidth=1.5, errorevery=15)
#         plot_indicator(system, w_means, ax, c)
    
#     def plot_indicator(system, indicator, ax, c=None, label=None):
#         target = indicator[0]
#         nearest = system.iloc[(system['#Coor'] -target).abs().argsort()[:1]]
#         near_true = nearest
#         x_val = near_true['#Coor']
#         y_val = near_true['Free']
#         ax.scatter(x_val, y_val, s=50, c=c, label=f'{label} {target:.2f} nm \u00B1 {indicator[1]:.2f} nm')
#         return None
    
    
#     def plot_free_energy(ax, system, indicator=None, title='Free Energy Profile', c=None, label=None):
#         df = system
#         ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Free'],
#                      yerr=df.loc[:, '+/-'], c=c, capsize=2.5, capthick=1.2,
#                      linewidth=1.5, errorevery=15)
#         if indicator is not None:
#             plot_indicator(system, indicator, ax, c, label=label)
#         plt.legend()    
                    
 

              
                
                
                
                
                

# class ComUmbrellaSampling(BaseUmbrellaSampling):
#     def __init__(self, file_dir, system, n_windows, xmin, xmax, com_list, ref_list):
#         super().__init__(file_dir, system)
#         self.f = Force()
#         self.obs = Observable()
#         self.file_dir = file_dir
#         self.n_windows = n_windows
#         self.xmin = xmin
#         self.xmax = xmax
#         self.com_list = com_list
#         self.ref_list = ref_list

#     def run_full_umbrella(self, eq_steps, xmax_steps, steps_per_conf, equlibration_steps, production_steps):
#         self.com_observable(print_every=1e4, name='com_distance.txt')
#         self.eq_process(eq_steps)
#         self.xmin_process(self.xmin_com_force_1, self.xmin_com_force_2, xmax_steps)
#         self.pull_process(self.pull_com_force_1, self.pull_com_force_2, self.n_windows, self.xmin, self.xmax, steps_per_conf)
#         self.equlibration_windows(self.n_windows)
        
#         manager = SimulationManager()
#         for sim, umbrella_force_1, umbrella_force_2 in zip(self.equlibration_sims, self.umbrella_forces_1, self.umbrella_forces_2):
#             sim.build(clean_build='force')
#             sim.add_force(umbrella_force_1)
#             sim.add_force(umbrella_force_2)
#             sim.add_observable(self.com_distance_observable)
#             sim.input_file({'steps':f'{equlibration_steps}', 'print_conf_interval':f'{equlibration_steps}', 'print_energy_every':f'{equlibration_steps}'})
#             manager.queue_sim(sim)
#         manager.worker_manager()
#         for sim in self.equlibration_sims: sim.sim_files.parse_current_files()
        
#         self.production_windows(self.n_windows)
#         for sim, umbrella_force_1, umbrella_force_2 in zip(self.production_sims, self.umbrella_forces_1, self.umbrella_forces_2):
#             sim.build(clean_build='force')
#             sim.add_force(umbrella_force_1)
#             sim.add_force(umbrella_force_2)
#             sim.add_observable(self.com_distance_observable)
#             sim.input_file({'steps':f'{production_steps}', 'print_conf_interval':f'{production_steps}', 'print_energy_every':f'{production_steps}'})
#             manager.queue_sim(sim)
#         manager.worker_manager()
#         for sim in self.production_sims: sim.sim_files.parse_current_files()
    
#     def gen_forces(self, xmin_stiff, pull_stiff, umbrella_stiff):
#         self.xmin_force(xmin_stiff)
#         self.pulling_force(pull_stiff)
#         self.umbrella_forces(umbrella_stiff)
    
#     def com_observable(self, print_every=1e4, name='com_distance.txt'):
#         self.com_distance_observable = self.obs.distance(
#             particle_1=self.com_list,
#             particle_2=self.ref_list,
#             print_every=f'{print_every}',
#             name=f'{name}'
#         )        
    
#     def xmin_force(self, stiff):
#         self.xmin_com_force_1 = self.f.com_force(
#             com_list=self.com_list,                        
#             ref_list=self.ref_list,                        
#             stiff=stiff,                    
#             r0=self.xmin,                       
#             PBC='1',                         
#             rate='0',
#         )       
#         self.xmin_com_force_2= self.f.com_force(
#             com_list=self.ref_list,                        
#             ref_list=self.com_list,                        
#             stiff=self.xmin,                    
#             r0=self.xmin,                       
#             PBC='1',                         
#             rate='0',
#         )
    
#     def pulling_force(self, stiff):
#         self.pull_com_force_1 = self.f.com_force(
#             com_list=self.com_list,                        
#             ref_list=self.ref_list,                        
#             stiff=stiff,                    
#             r0=self.xmin,                       
#             PBC='1',                         
#             rate='0',
#         )        
#         self.pull_com_force_2 = self.f.com_force(
#             com_list=self.ref_list,                        
#             ref_list=self.com_list,                        
#             stiff=stiff,                    
#             r0=self.xmin,                       
#             PBC='1',                         
#             rate='0',
#         )
    
#     def umbrella_forces(self, stiff):
#         x_range = np.linspace(self.xmin, self.xmax, (self.n_windows + 1))[1:]

#         self.umbrella_forces_1 = []
#         self.umbrella_forces_2 = []
        
#         for x_val in x_range:   
#             self.umbrella_force_1 = self.f.com_force(
#                 com_list=self.com_list,                        
#                 ref_list=self.ref_list,                        
#                 stiff=f'{stiff}',                    
#                 r0=f'{x_val}',                       
#                 PBC='1',                         
#                 rate='0',
#             )        
#             self.umbrella_forces_1.append(self.umbrella_force_1)
            
#             self.umbrella_force_2= self.f.com_force(
#                 com_list=self.ref_list,                        
#                 ref_list=self.com_list,                        
#                 stiff=f'{stiff}',                    
#                 r0=f'{x_val}',                       
#                 PBC='1',                          
#                 rate='0',
#             )
#             self.umbrella_forces_2.append(self.umbrella_force_2)

            

    
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

   
            
            
# def sep_conf(last_conf, traj, sim_dir):
#     """
#     Separate the trajectory file into separate files for each umbrella window.
#     :param last_conf: path to last configuration file of pulling simulations
#     :param traj: path to trajectory file of pulling simulations
#     :param sim_dir: directory to save the separated files
#     :return: None
#     """
#     # Create base umbrella simulation directory if it does not exist
#     if not exists(sim_dir):
#         os.mkdir(sim_dir)
#     # get number of lines in a singe oxDNA configuration
#     num_lines = sum(1 for line in open(last_conf))
#     with open(traj, 'r') as f:
#         # read trajectory file
#         file = f.readlines()
#         # get the number of configurations in the trajectory file
#         n_confs = len(file) // num_lines
#         for i in range(n_confs):
#             # create a new window(dir) for each configuration
#             c_dir = join(sim_dir, str(i))
#             if not exists(c_dir):
#                 os.mkdir(c_dir)
#             # Write the configuration to the new window
#             with open(join(c_dir, f'conf_{i}.dat'), 'w') as f:
#                 for j in range(num_lines):
#                     f.write(file[i * num_lines + j])
#     return n_confs


# def get_input(**input_dict):
#     """
#     Create input file object for umbrella simulation using the input_dict dictionary as input file Parameters in oxpy context.
#     :param input_dict: dictionary of input parameters
#     :return: my_input: input file as oxpy input object
#     """
#     with oxpy.Context():
#         my_input = oxpy.InputFile()
#         # assign input parameters to input object
#         for k, v in input_dict.items():
#             if v:
#                 my_input[k] = v
#     return my_input


# def get_windows_per_gpu(n_gpus, n_confs):
#     """
#     Calculate the optimal number of windows per gpu
#     :param n_gpus:
#     :return:
#     """
#     round = n_confs // n_gpus
#     remainder = n_confs % n_gpus
#     w_p_gpu = []
#     for i in range(n_gpus):
#         if remainder != 0:
#             w_p_gpu.append(round + 1)
#             remainder -= 1
#         else:
#             w_p_gpu.append(round)
#     w_p_gpu.sort()
#     return w_p_gpu


# def write_production_run_file(run_file, windows_per_gpu, sim_dir):
#     # Write hpc run file to each window
#     with open(run_file, 'r') as f:
#         lines = f.readlines()
#         window = 0
#         for gpu_set in windows_per_gpu:
#             for file in range(gpu_set):
#                 with open(join(sim_dir, str(window), 'run.sh'), 'w') as r:
#                     win = window + 1
#                     for line in lines:
#                         if 'job-name' in line:
#                             r.write(f'#SBATCH --job-name="{window}"\n')
#                         else:
#                             r.write(line)
#                     if file < (gpu_set - 1):
#                         r.write(f'cd {join(sim_dir, str(win))}\nsbatch run.sh')
#                 window += 1
#     return None


# def run_production_slurm_files(n_gpus, n_windows, sim_dir):
#     windows_per_gpu = get_windows_per_gpu(n_gpus, n_windows)
#     run_dirs = [0]
#     for gpu_set in windows_per_gpu:
#         run_dirs.append(run_dirs[-1] + gpu_set)
#     run_dirs.pop()
#     for i in run_dirs:
#         os.chdir(join(f"{sim_dir}", str(i)))
#         os.system("sbatch run.sh")
        
