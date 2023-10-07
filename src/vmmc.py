from oxdna_simulation import Simulation, Force, Observable, SimulationManager, OxpyRun, GenerateReplicas, Analysis
import os
import shutil
import queue
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import scienceplots
from scipy.stats import norm, t, sem

def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b


class VirtualMoveMonteCarlo(Simulation):
    def __init__(self, file_dir, sim_dir):
        super().__init__(file_dir, sim_dir)
        self.vmmc_obs = VmmcObservables(self)
        self.analysis = VmmcAnalysis(self)
        self.oxpy_run = VmmcOxpyRun(self)
    
    def build(self, p1, p2, clean_build=False, pre_defined_weights=None):
        """
        Build dat, top, and input files in simulation directory.
        
        Parameters:
            clean_build (bool): If sim_dir already exsists, remove it and then rebuild sim_dir
        """
        if os.path.exists(self.sim_dir):
            #print(f'Exisisting simulation files in {self.sim_dir.split("/")[-1]}')            
            if clean_build == True:               
                answer = input('Are you sure you want to delete all simulation files? Type y/yes to continue or anything else to return (use clean_build=str(force) to skip this message)')
                if (answer == 'y') or (answer == 'yes'):
                    shutil.rmtree(f'{self.sim_dir}/')
                else:
                    print('Remove optional argument clean_build and rerun to continue')
                    return None           
            elif clean_build == 'force':                    
                    shutil.rmtree(self.sim_dir)                
        self.build_sim.build_sim_dir()
        self.build_sim.build_dat_top()
        self.build_sim.build_input()
        self.vmmc_input()
        self.build_vmmc_op_file(p1, p2)
        self.build_vmmc_weight_file(p1, p2, pre_defined_weights)
    
        self.sim_files.parse_current_files()
    
    
    def build_vmmc_op_file(self, p1, p2):
        p1 = p1.split(',')
        p2 = p2.split(',')
        # print('Please ensure that nuclotide indexes are aligned such that the first nuc in one string binds to the first nuc in the other string, the seconds binds to the second, ect otherwise the result will be incorrect')
        with open(os.path.join(self.sim_dir,"vmmc_op.txt"), 'w') as f:
            f.write("{\norder_parameter = bond\nname = all_native_bonds\n")
            
            for i, (nuc1, nuc2) in enumerate(zip(p1, p2)):
                f.write(f'pair{i} = {nuc1}, {nuc2}\n')

        with open(os.path.join(self.sim_dir,"vmmc_op.txt"), 'a') as f:
            f.write("}\n")
        return None
    
    def build_vmmc_weight_file(self, p1, p2, pre_defined_weights):
        p1 = p1.split(',')
        with open(os.path.join(self.sim_dir,'wfile.txt'), 'w') as f:
            for idx in range(len(p1) + 1):
                if pre_defined_weights is not None:
                    f.write(f'{idx} {pre_defined_weights[idx]}\n')
                else:
                    f.write(f'{idx} 1\n')    
    
    def build_com_hb_observable(self, p1, p2):
        self.vmmc_obs.com_distance_observable(p1, p2, print_every=1e3)
        self.build_sim.build_hb_list_file(p1, p2)
        self.vmmc_obs.hb_list_observable(print_every=1e3)
        
        for observable in self.vmmc_obs.observables_list:
            self.add_observable(observable)
    
    def vmmc_input(self):
        vmmc_parameters = {
            'backend':'CPU',
            'sim_type':'VMMC',
            'check_energy_every':'100000',
            'check_energy_threshold':'1.e-4',
            'delta_translation':'0.1',
            'delta_rotation':'0.2',
            'umbrella_sampling':'1',
            'op_file':'vmmc_op.txt',
            'weights_file':'wfile.txt',
            'extrapolate_hist':'34C, 36C, 38C, 40C, 42C, 44C, 46C, 48C, 50C, 52C, 54C, 56C, 58C, 60C, 62C, 64C, 66C, 68C, 70C',
            'maxclust':'12',
            'small_system':'1',
            'last_hist_file':'last_hist.dat',
            'traj_hist_file':'traj_hist.dat'
        }
        self.input_file(vmmc_parameters)
            
class VmmcOxpyRun(OxpyRun):
    """Automatically runs a built oxDNA simulation using oxpy within a subprocess"""
    def __init__(self, sim):
        super().__init__(sim)
        # self.sim = sim
        # self.sim_dir = sim.sim_dir
        # self.my_obs = {}
        
    def run(self, subprocess=True, continue_run=False, verbose=True, log=True, join=False, custom_observables=None):
        """ Run oxDNA simulation using oxpy in a subprocess.
        
        Parameters:
            subprocess (bool): If false run simulation in parent process (blocks process), if true spawn sim in child process.
            continue_run (number): If False overide previous simulation results. If True continue previous simulation run.
            verbose (bool): If true print directory of simulation when run.
            log (bool): If true print a log file to simulation directory.
            join (bool): If true block main parent process until child process has terminated (simulation finished)
        """
        self.subprocess = subprocess
        self.verbose = verbose
        self.continue_run = continue_run
        self.log = log
        self.join = join
        self.custom_observables = custom_observables
        
        if continue_run is not False:
            self.sim.input_file({'init_hist_file':self.sim.input.input['last_hist_file']})
        if self.verbose == True:
            print(f'Running: {self.sim_dir.split("/")[-1]}')
        if self.subprocess:
            self.spawn(self.run_complete)
        else:
            self.run_complete()     
            
            
class VmmcObservables:
    def __init__(self, base_vmmc):
        self.obs = Observable()
        self.base_vmmc = base_vmmc
        self.observables_list = []
        
    
    def com_distance_observable(self, com_list, ref_list,  print_every=1e4, name='com_distance.txt'):
        """ Build center of mass observable"""
        obs = Observable()
        com_observable = self.obs.distance(
            particle_1=com_list,
            particle_2=ref_list,
            print_every=f'{print_every}',
            name=f'{name}',
            PBC='1'
        )  
        self.observables_list.append(com_observable)
        
    def hb_list_observable(self, print_every=1e4, name='hb_observable.txt', only_count='true'):
        """ Build center of mass observable"""
        hb_obs = self.obs.hb_list(
            print_every='1e3',
            name='hb_observable.txt',
            only_count='true'
           )
        self.observables_list.append(hb_obs)
        
        
class VmmcReplicas(GenerateReplicas):
    def __init__(self):
        self.prev_num_bins = None
        self.prev_confidence_level = None
        self.replica_histograms = None
        self.all_free_energies = None
        self.sem_free_energy = None
        
    def multisystem_replica(self, systems, n_replicas_per_system, file_dir_list, sim_dir_list):
        """
        Create simulation replicas, with across multiple systems with diffrent inital files
        
        Parameters:
            systems (list): List of strings, where the strings are the name of the directory which will hold the inital files
            n_replicas_per_system (int): number of replicas to make per system
            file_dir_list (list): List of strings with path to intial files
            sim_dir_list (list): List of simulation directory paths
        """
        self.systems = systems
        self.n_replicas_per_system = n_replicas_per_system
        self.file_dir_list = file_dir_list
        self.sim_dir_list = sim_dir_list
        
        replicas = range(n_replicas_per_system)
        sim_rep_list = []
        for sys in sim_dir_list:
            for rep in replicas:
                sim_rep_list.append(f'{sys}_{rep}')
        q1 = queue.Queue()
        for sys in sim_rep_list:
            q1.put(sys)
        sim_list = []
        
        for file_dir in file_dir_list:
            for _ in range(len(replicas)):
                sim_dir = q1.get()
                sim_list.append(VirtualMoveMonteCarlo(file_dir, sim_dir))
        q2 = queue.Queue()
        for sim in sim_list:
            q2.put(sim)
        
        self.sim_list = sim_list
        self.queue_of_sims = q2



    def plot_mean_free_energy_with_error_bars(self, num_bins=50, confidence_level=0.95, ax=None, label=None, errorevery=10):
        """Plot the mean free energy landscape with confidence intervals.
        
        Parameters:
            num_bins (int): Number of bins for histogram.
            confidence_level (float): Confidence level for confidence intervals.
            ax (matplotlib axis, optional): Axis on which to plot the graph.
        """
        recompute = (
            self.prev_num_bins != num_bins or 
            self.prev_confidence_level != confidence_level or 
            self.replica_histograms is None or 
            self.all_free_energies is None or 
            self.sem_free_energy is None
        )

        if recompute:
            self.collect_replica_histograms(num_bins=num_bins)
            self.calculate_individual_free_energies()
            self.calculate_sem_free_energy()

        # Update previous values
        self.prev_num_bins = num_bins
        self.prev_confidence_level = confidence_level
        # Step 4: Calculate Z-score for the given confidence level
        z_score = norm.ppf(1 - (1 - confidence_level) / 2)
        
        # Step 5: Calculate the confidence interval
        confidence_interval = z_score * self.sem_free_energy
        
        # Step 6: Plot mean and error
        min_val = float(self.sim_list[0].analysis.com_distance['com_distance'].min())
        max_val = float(self.sim_list[0].analysis.com_distance['com_distance'].max())
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean_free_energy = np.nanmean(self.all_free_energies, axis=0)
        if ax is None:
            fig, ax = plt.subplots(dpi=200, figsize=(5.5, 4.5))
        if label is None:
            label = 'VMMC free energy made discrete'
        with plt.style.context(['science', 'no-latex', 'bright']):
            ax.errorbar(bin_centers * 0.85, mean_free_energy, yerr=confidence_interval, fmt='-', capsize=2.5, capthick=1.2, linewidth=1.5, label=label, errorevery=errorevery)
            # ax.fill_between(bin_centers * 0.85, mean_free_energy - confidence_interval, mean_free_energy + confidence_interval, color='gray', alpha=0.5)
            # ax.set_xlabel('COM Distance')
            # ax.set_ylabel('Free Energy')
            # ax.set_title(f'Mean Free Energy Landscape with {int(confidence_level*100)}% Confidence Intervals')

        
    def calculate_individual_free_energies(self):
        self.all_free_energies = []
        for idx,histogram in enumerate(self.replica_histograms):
            # Check for empty histogram
            if histogram.size == 0:
                print("Empty histogram encountered.")
                continue
        
            # Check for all zeros
            if np.all(histogram == 0):
                print("Histogram contains only zeros.")
                continue
        
            # Replace zeros with non-zero minimum
            non_zero_min = histogram[histogram > 0]
            if non_zero_min.size > 0:
                min_val = np.nanmin(non_zero_min)
                histogram[histogram == 0] = min_val
            else:
                # print(histogram)
                # print(idx)
                print("No non-zero minimum value found.")

            # Calculate free energy
            free_energy = -np.log(histogram)
            
            # Shift so that minimum free energy is zero
            min_free_energy = np.min(free_energy)
            free_energy -= min_free_energy
            
            self.all_free_energies.append(free_energy)
        
    def calculate_sem_free_energy(self):
        # Convert list to NumPy array for easier calculations
        all_free_energies_array = np.array(self.all_free_energies)
        
        # Calculate SEM for each bin across all free energy profiles
        self.sem_free_energy = np.nanstd(self.all_free_energies, axis=0) / np.sqrt(len(self.all_free_energies))
       
        
    def collect_replica_histograms(self, num_bins=50):
        # Initialize a list to store histograms from each replica
        self.replica_histograms = []
        
        for sim in self.sim_list:
            sim.analysis.read_files()
            try:
                sim.analysis.calculate_weighted_histogram(num_bins=num_bins)
                self.replica_histograms.append(sim.analysis.weighted_histogram)
            except:
                pass
        
        # Convert list of arrays to a 2D numpy array for easier analysis
        # self.replica_histograms = np.array(self.replica_histograms)

    def analyze_histogram_convergence_and_errors(self):
        # Calculate the mean, standard deviation, and SEM across replicas
        self.mean_histogram = np.nanmean(self.replica_histograms, axis=0)
        self.std_histogram = np.nanstd(self.replica_histograms, axis=0)
        self.sem_histogram = self.std_histogram / np.sqrt(len(self.sim_list))

    def plot_histogram_convergence_and_errors(self, num_bins=50):
        # Calculate bin centers for plotting
        min_val = float(self.sim_list[0].analysis.com_distance['com_distance'].min())
        max_val = float(self.sim_list[0].analysis.com_distance['com_distance'].max())
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plotting
        plt.figure(figsize=(12, 8))
        plt.plot(bin_centers, self.mean_histogram, label='Mean across replicas')
        plt.fill_between(bin_centers, 
                         self.mean_histogram - self.sem_histogram, 
                         self.mean_histogram + self.sem_histogram, 
                         color='gray', alpha=0.5, label='SEM')
        plt.xlabel('COM Distance')
        plt.ylabel('Weighted Probability')
        plt.title('Histogram Convergence and Error Analysis')
        plt.legend()
        plt.show()
        
    def statistical_analysis_and_plot(self, confidence_level=0.999):
        """
        Perform statistical analysis over all simulation replicas and plot the results.
        """
        for sim in self.sim_list:
            sim.analysis.read_vmmc_op_data()
            sim.analysis.calculate_sampling_and_probabilities()
            sim.analysis.calculate_and_estimate_melting_profiles()
        
        wt_prod = np.array([sim.statistics['wt_prob'].values for sim in self.sim_list])
        wt_free = np.array([sim.statistics['wt_free'].values for sim in self.sim_list])
        sampling_percent = np.array([sim.statistics['sampling_percent'].values for sim in self.sim_list])
        
        temp_columns_prob = [[col for col in sim.statistics.columns if '_prob' in col and 'wt_occ' in col] for sim in self.sim_list]
        heat_map = [sim.statistics[columns].values for sim, columns in zip(self.sim_list, temp_columns_prob)]
        
        x_fit = sim.analysis.x_fit
        y_fit = np.array([sim.analysis.y_fit for sim in self.sim_list])
        inverted_finfs = np.array([sim.analysis.inverted_finfs for sim in self.sim_list])
        tm = np.array([sim.Tm for sim in self.sim_list])
        temperatures = self.sim_list[0].analysis.temperatures
        
        df = len(tm) - 1
        
        wt_prob_mean = np.mean(wt_prod, axis=0)
        wt_prob_sem = sem(wt_prod, axis=0)
        wt_prob_ci = t.interval(confidence_level, df, loc=wt_prob_mean, scale=wt_prob_sem)
        
        wt_free_mean = np.mean(wt_free, axis=0)
        wt_free_sem = sem(wt_free, axis=0)
        wt_free_ci = t.interval(confidence_level, df, loc=wt_free_mean, scale=wt_free_sem)

        sampling_percent_mean = np.mean(sampling_percent, axis=0)
        sampling_percent_sem = sem(sampling_percent, axis=0)
        sampling_percent_ci = t.interval(confidence_level, df, loc=sampling_percent_mean, scale=sampling_percent_sem)
        
        heat_map_mean = np.mean(heat_map, axis=0)

        y_fit_mean = np.mean(y_fit, axis=0)
        y_fit_sem = sem(y_fit, axis=0)
        y_fit_ci = t.interval(confidence_level, df, loc=y_fit_mean, scale=y_fit_sem)

        inverted_finfs_mean = np.mean(inverted_finfs, axis=0)
        inverted_finfs_sem = sem(inverted_finfs, axis=0)
        inverted_finfs_ci = t.interval(confidence_level, df, loc=inverted_finfs_mean, scale=inverted_finfs_sem)

        tm_mean = np.mean(tm)
        tm_sem = sem(tm)
        tm_ci = t.interval(confidence_level, df, loc=tm_mean, scale=tm_sem)

        n_bonds = list(range(len(sampling_percent_mean)))
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        axes[0, 0].plot(n_bonds, wt_prob_mean)
        axes[0, 0].fill_between(range(9), wt_prob_ci[0], wt_prob_ci[1], interpolate=True, color='gray', alpha=0.5)
        axes[0, 0].set_xlabel('Number of Hydrogen Bonds')
        axes[0, 0].set_ylabel('wt_prob')
        
        
        axes[0, 1].plot(n_bonds, wt_free_mean)
        axes[0, 1].fill_between(range(len(sampling_percent_mean)), wt_free_ci[0], wt_free_ci[1], interpolate=True, color='gray', alpha=0.5)
        axes[0, 1].set_xlabel('Number of Hydrogen Bonds')
        axes[0, 1].set_ylabel('wt_free')
        
        axes[1, 0].bar(n_bonds, sampling_percent_mean)
        # axes[1, 0].fill_between(range(9), sampling_percent_ci[0], sampling_percent_ci[1], color='gray', alpha=0.5)
        axes[1, 0].set_xlabel('Number of Hydrogen Bonds')
        axes[1, 0].set_ylabel('Probability')
        
        extent = [min(temperatures), max(temperatures), min(n_bonds), max(n_bonds)]
        im = axes[1, 1].imshow(heat_map_mean, extent=extent, cmap='viridis', aspect='auto')
        axes[1, 1].set_title('Heatmap of wt_prob across Temperatures')
        axes[1, 1].set_xlabel('Temperature')
        axes[1, 1].set_ylabel('Number of Hydrogen Bonds')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.figure()
        plt.scatter(temperatures, inverted_finfs_mean, marker='o', label='Data Mean')
        plt.plot(x_fit, y_fit_mean, linestyle='--', linewidth=2, label='Sigmoid Fit')
        plt.fill_between(temperatures, inverted_finfs_ci[0], inverted_finfs_ci[1], interpolate=True, color='gray', alpha=0.5)
        plt.axvline(x=tm_mean, color='r', linestyle='--', linewidth=2, label=f'Tm = {tm_mean:.2f} \u00b1 {tm_ci[1] - tm_mean:.2f} °C')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Fraction of ssDNA')
        plt.title(f'Melting Profile')
        
        # Set y-axis limits
        plt.ylim(0, 1.1)
        
        plt.legend()
        plt.grid(True)
                
class VmmcAnalysis(Analysis):
    """ Methods used to interface with oxDNA simulation in jupyter notebook (currently in work)"""
    def __init__(self, simulation):
        """ Set attributes to know all files in sim_dir and the input_parameters"""
        self.sim = simulation
        self.sim_dir = self.sim.sim_dir
        self.sim_files = simulation.sim_files
        self.sim.melting_profiles = None
        self.com_distance = None
        self.hb_observable = None
        self.weights = None
        self.weighted_histogram = None

        
    def read_files(self):
        com_distance_path = os.path.join(self.sim_dir, 'com_distance.txt')
        hb_observable_path = os.path.join(self.sim_dir, 'hb_observable.txt')
        wfile_path = os.path.join(self.sim_dir, 'wfile.txt')
    
        self.com_distance = pd.read_csv(com_distance_path, header=None, names=['com_distance'])
        self.hb_observable = pd.read_csv(hb_observable_path, header=None, names=['hb_observable'])
        self.weights = pd.read_csv(wfile_path, header=None, names=['index', 'weight'], delim_whitespace=True)
        
        # Convert 'weight' to float if it's not already
        if self.weights['weight'].dtype == 'object':
            self.weights['weight'] = self.weights['weight'].apply(lambda x: x.rstrip('.') if x.endswith('.') else x).astype(float)

    def calculate_weighted_histogram(self, num_bins=50):
        # Create an empty histogram
        self.weighted_histogram = np.zeros(num_bins)
        
        # Ensure min and max are scalar values
        min_val = float(self.com_distance['com_distance'].min())
        max_val = float(self.com_distance['com_distance'].max())
        
        # Create bin edges
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        
        # Create a mapping of hb_observable to weight for faster lookup
        weight_mapping = self.weights.set_index('index')['weight'].to_dict()
        
        # Vectorized weight lookup
        weights_vector = self.hb_observable['hb_observable'].map(weight_mapping).values
        
        # Vectorized bin index calculation
        bin_indices = np.digitize(self.com_distance['com_distance'], bin_edges) - 1
        
        # Clip bin indices to be within bounds
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # Calculate the weighted histogram
        np.add.at(self.weighted_histogram, bin_indices, 1 / weights_vector)
        
        # Normalize the histogram
        self.weighted_histogram /= np.sum(self.weighted_histogram)
        
        # Adding a small constant to avoid log(0)
        epsilon = 1e-15
        self.free_energy = -np.log(self.weighted_histogram + epsilon)
        
        # Shift the free energy profile so that the minimum value is 0
        min_free_energy = np.min(self.free_energy)
        self.free_energy -= min_free_energy



    def plot_weighted_histogram(self, n_bins=50, label=None, ax=None):
        self.read_files()
        self.calculate_weighted_histogram(num_bins=n_bins)
        
        # Create a new figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate the center of each bin
        min_val = float(self.com_distance['com_distance'].min())
        max_val = float(self.com_distance['com_distance'].max())
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot on the provided axes
        ax.plot(bin_centers * 0.85, self.free_energy, label=label)
        
        # If ax was not provided, finalize the plot
        ax.set_xlabel('COM Distance')
        ax.set_ylabel('Free Energy')
        ax.set_title('Free Energy Landscape')

    def last_hist_analysis(self):
        self.read_vmmc_op_data()
        self.calculate_sampling_and_probabilities()
        self.plot_statistics()
        self.plot_melting_profiles()        
        
    def read_vmmc_op_data(self):
        self.sim_files.parse_current_files()
        
        # Initialize variables to store metadata and data
        simulation_time = None
        temperatures = []
        data = []
        
        # Read the file line by line
        with open(self.sim_files.last_hist, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    # Parse metadata line
                    metadata_parts = line.strip().split(";")
                    simulation_time = int(metadata_parts[0].split("=")[1].strip())
                    sim_temp_units = list(map(float, metadata_parts[1].split(":")[1].strip().split()))
                    
                    # Convert simulation temperature units to Celsius
                    temperatures = [(temp_unit * 3000) - 273.15 for temp_unit in sim_temp_units]
                else:
                    # Parse data lines and convert to float
                    row = list(map(float, line.strip().split()))
                    data.append(row)

        # Create DataFrame from data
        df = pd.DataFrame(data)

        # Rename columns with lowercase names
        column_names = ["h_bonds", "unwt_occ", "wt_occ"]
        column_names += [f"wt_occ_{temp:.1f}C" for temp in temperatures]
        df.columns = column_names
        
        self.sim.simulation_time = simulation_time
        self.sim.vmmc_df = df

    def calculate_sampling_and_probabilities(self):
        """Calculate the sampling percentage, probability, and -log(probability) for each occurrence."""
        
        self.sim.statistics = pd.DataFrame()
        
        # Calculate the total unweighted occurrences in the simulation
        total_unwt_occ = self.sim.vmmc_df['unwt_occ'].sum()
        
        # Calculate the sampling percentage for each state
        self.sim.statistics['sampling_percent'] = (self.sim.vmmc_df['unwt_occ'] / total_unwt_occ) * 100
        
        # Calculate the total weighted occurrences in the simulation
        total_wt_occ = self.sim.vmmc_df['wt_occ'].sum()
        
        # Calculate the probability for each state
        self.sim.statistics['wt_prob'] = self.sim.vmmc_df['wt_occ'] / total_wt_occ
        # Avoid log(0) by replacing zeros
        epsilon = 1e-15
        # self.sim.statistics['wt_prob'][self.sim.statistics['wt_prob'] == 0] = epsilon
        
        # Calculate the -log(probability) for each state
        self.sim.statistics['wt_free'] = -np.log(self.sim.statistics['wt_prob'] + epsilon)
        # Shift the free energy values so that the lowest is zero
        min_wt_free = self.sim.statistics['wt_free'].min()
        self.sim.statistics['wt_free'] -= min_wt_free
        
        # Calculate probabilities and -log(probabilities) for extrapolated temperatures
        temp_columns = [col for col in self.sim.vmmc_df.columns if col.startswith("wt_occ_")]
        for col in temp_columns:
            total_temp_occ = self.sim.vmmc_df[col].sum()
            prob_col = f"{col}_prob"
            neglog_prob_col = f"{col}_free"
            
            self.sim.statistics[prob_col] = self.sim.vmmc_df[col] / total_temp_occ
            # self.sim.statistics[prob_col][self.sim.statistics[prob_col] == 0] = epsilon
            self.sim.statistics[neglog_prob_col] = -np.log(self.sim.statistics[prob_col] + epsilon)
            
            # Shift the free energy values so that the lowest is zero for each temperature
            min_temp_free = self.sim.statistics[neglog_prob_col].min()
            self.sim.statistics[neglog_prob_col] -= min_temp_free
            
    def plot_statistics(self):
        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        
        # Line plot for wt_prob
        self.sim.statistics['wt_prob'].plot(ax=axes[0, 0], title='Weighted Probability (wt_prob)', color='b')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('wt_prob')
        
        # Line plot for wt_free
        self.sim.statistics['wt_free'].plot(ax=axes[0, 1], title='Negative Log Probability (wt_free)', color='r')
        axes[0, 1].set_xlabel('Index')
        axes[0, 1].set_ylabel('wt_free')
        
        # Bar plot for sampling_percent
        self.sim.statistics['sampling_percent'].plot(kind='bar', ax=axes[1, 0], title='Sampling Percentage', color='g')
        axes[1, 0].set_xlabel('Index')
        axes[1, 0].set_ylabel('sampling_percent')
        
        # Heatmap for wt_prob and wt_free across temperatures
        temp_columns_prob = [col for col in self.sim.statistics.columns if '_prob' in col and 'wt_occ' in col]
        temp_columns_free = [col for col in self.sim.statistics.columns if '_free' in col and 'wt_occ' in col]
        
        im = axes[1, 1].imshow(self.sim.statistics[temp_columns_prob].values, cmap='viridis', aspect='auto')
        axes[1, 1].set_title('Heatmap of wt_prob across Temperatures')
        axes[1, 1].set_xlabel('Temperature')
        axes[1, 1].set_ylabel('Index')
        plt.colorbar(im, ax=axes[1, 1])
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
    def calculate_and_estimate_melting_profiles(self):
        """
        Calculate the melting profiles and estimate the melting temperature (Tm).
        """
        # Initialize an empty DataFrame to store the melting profiles
        self.sim.melting_profiles = pd.DataFrame()
    
        # Initialize list to store finite-size-effect corrected yields (finfs)
        finfs = []
        temperatures = []  # Initialize list to store temperatures
    
        # Loop through each temperature column in the vmmc_df DataFrame
        for col in self.sim.vmmc_df.columns:
            if col.startswith("wt_occ_"):
                # Extract temperature from column name
                try:
                    temp = float(col.split('_')[-1].replace('C', ''))
                except ValueError:
                    continue
    
                # Calculate the ratio of bound to unbound states for this temperature
                bound_states = self.sim.vmmc_df[col][self.sim.vmmc_df['h_bonds'] > 0].sum()
                unbound_states = self.sim.vmmc_df[col][self.sim.vmmc_df['h_bonds'] == 0].sum()
    
                # Calculate the melting ratio and finf
                ratio = bound_states / unbound_states if unbound_states != 0 else np.nan
                finf = 1. + 1. / (2. * ratio) - math.sqrt((1. + 1. / (2. * ratio)) ** 2 - 1.)
    
                # Add this ratio and finf to their respective data structures
                self.sim.melting_profiles[col] = [ratio]
                finfs.append((temp, finf))
                temperatures.append(temp)  # Add temperature to the list
    
        # Check if finfs is empty
        if not finfs:
            print("Warning: No finite-size-effect corrected yields (finfs) calculated.")
            return
    
        # Store finfs and temperatures as instance variables
        self.finfs = [f for _, f in sorted(finfs, key=lambda x: x[0])]
        self.temperatures = sorted(temperatures)
    
        # Estimate Tm based on finfs
        self.sim.Tm = self._get_Tm(self.temperatures, self.finfs)
        # print(f"Estimated Melting Temperature (Tm) = {self.sim.Tm} °C")
        
        # Invert the finfs to get the fraction of ssDNA
        self.inverted_finfs = [1 - finf for finf in self.finfs]
    
        # Fit the sigmoid function to the inverted data
        p0 = [max(self.inverted_finfs), np.median(self.temperatures), 1, min(self.inverted_finfs)]  # initial guesses for L, x0, k, b
        popt, _ = curve_fit(sigmoid, self.temperatures, self.inverted_finfs, p0, method='dogbox')
    
        # Generate fitted data
        self.x_fit = np.linspace(min(self.temperatures), max(self.temperatures), 500)
        self.y_fit = sigmoid(self.x_fit, *popt)
        
        idx = np.argmin(np.abs(self.y_fit - 0.5))
        self.sim.Tm = self.x_fit[idx]
        

    def _get_Tm(self, temps, finfs):
        """
        Helper function to estimate Tm.
        """
        x = finfs.copy()
        y = temps.copy()
        x.reverse()
        y.reverse()
        xin = np.arange(0.1, 1., 0.1)
        f = np.interp(xin, np.array(x), np.array(y))
        return f[4]

    def plot_melting_profiles(self):
        # Ensure melting profiles are calculated
        self.calculate_and_estimate_melting_profiles()
    
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(self.temperatures, self.inverted_finfs, marker='o', label='Data')
        plt.plot(self.x_fit, self.y_fit, linestyle='--', linewidth=2, label='Sigmoid Fit')
        
        # Add a vertical line at the melting temperature
        plt.axvline(x=self.sim.Tm, color='r', linestyle='--', linewidth=2, label=f'Tm = {self.sim.Tm:.2f} °C')
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Fraction of ssDNA')
        plt.title(f'Melting Profile')
        
        # Set y-axis limits
        plt.ylim(0, 1.1)
        
        plt.legend()
        plt.grid(True)
        plt.show()

# I now want to read the com_distance.txt, hb_observable.txt, and wfile.txt files within self.sim_dir (which is the abspath to the directory where the simulation is being run) and then I want to create a weighted histogram using the weights in the weight file. I can because I have the time series of the number of hydrogen bonds and of the center of mass distance. But, the occurance of the center of mass distances is currently biased. The center of mass distance is according to the bias of the number of hydrogen bonds associated with the distance. For example is head of the 
# com_distance.txt file is:
#         0.4213
#         0.3247
#         0.4286
#         0.3116
#         0.4170
#         0.4395
#         0.3621
#         0.4143
#         0.4748
#         0.4640
#         0.5616
#         0.3931
#         0.5516
#         0.4508
#         0.9298
#         0.8375
# and the head of the the hb_observable.txt file is:
# 8
# 8
# 8
# 8
# 7
# 6
# 8
# 8
# 6
# 6
# 7
# 4
# 5
# 6
# 3
# 5

# and the wfile.txt is:
# 0 8.
# 1 16204.
# 2 1882.94.
# 3 359.746.
# 4 52.5898.
# 5 15.0591.
# 6 7.21252.
# 7 2.2498.
# 8 2.89783.

# Then I can unbias the probability of certain com distances by using the assosiated weight. Specifically, when creating a histogram you can imagine pulling com_distance values from a queue, and then "tossing" that value into its respective bin and adding +1. To unbias the probability, I would have to toss the value into the bin and add 1 / weight.