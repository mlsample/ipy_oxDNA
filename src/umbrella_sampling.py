from oxdna_simulation import Simulation, Force, Observable, SimulationManager
from wham_analysis import *
import multiprocessing as mp
import os
from os.path import join, exists
import numpy as np
import shutil
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import scienceplots


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
            
    def wham_run(self, wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot):
        self.wham.run_wham(wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot)
        self.wham.to_si(n_bins)
        self.wham.w_mean()
        try:
            self.wham.bootstrap_w_mean_error()
        except:
            self.standard_error = 'failed'
    
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
            try:
                self.wham.bootstrap_w_mean_error()
            except:
                self.standard_error = 'failed'
            

class ComUmbrellaSampling(BaseUmbrellaSampling):
    def __init__(self, file_dir, system):
        super().__init__(file_dir, system)
        self.observables_list = []
        
    def build_equlibration_runs(self, simulation_manager,  n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                                observable=False, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False,
                                protein=None, force_file=None):
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
    def __init__(self, file_dir, system):
        super().__init__(file_dir, system)
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
    def __init__(self, file_dir, system):
        super().__init__(file_dir, system)
        
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
    
    
class UmbrellaBuild:
    def __init__(self, base_umbrella):
        pass
    
    def build(self, sims, input_parameters, forces_list, observables_list,
              observable=False, sequence_dependant=False, cms_observable=False, protein=None, force_file=None):
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
    
    
    def plot_free_energy(self, ax=None, title='Free Energy Profile', label=None,errorevery=1):
        if ax is None:
            ax = self.plt_fig()
        if label is None:
            label = self.base_umbrella.system
        # if c is None:
        #     c = '#00429d'
        indicator = [self.base_umbrella.mean, self.base_umbrella.standard_error]
        df = self.base_umbrella.free
        try:
            ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Free'],
                     yerr=df.loc[:, '+/-'], capsize=2.5, capthick=1.2,
                     linewidth=1.5, errorevery=errorevery, label=f'{label} {indicator[0]:.2f} nm \u00B1 {indicator[1]:.2f} nm')
            if indicator is not None:
                self.plot_indicator(indicator, ax, label=label)
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

            