import io
import os
import numpy as np
import contextlib
import sys
from copy import deepcopy
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal, norm, t
import traceback

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


class PymbarAnalysis:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
        self.hist = PymbarHistograming(self)

    
    def run_mbar_fes(self, reread_files=False, sim_type='prod', uncorrelated_samples=False, restraints=False, force_energy_split=False, verbose=True):
            
        u_kln, N_k = self._setup_input_data(reread_files=reread_files, sim_type=sim_type, 
                                            restraints=restraints, force_energy_split=force_energy_split,
                                            uncorrelated_samples=uncorrelated_samples)
        
        mbar_options = {'solver_protocol':'jax'}
        
        if verbose:
            print("Running FES calculation...")
        with suppress_output():
            self.basefes = pymbar.FES(u_kln, N_k, mbar_options=mbar_options, verbose=verbose, timings=True)
        if verbose:
            print("MBAR calculation complete.")

    def pymbar_convergence_free_energy_curves(
        self,
        convergence_splits: int,
        op_strings=['com_distance', 'hb_list'],
        subsampling=1,
        n_bins=50,
        temp_range=np.arange(30, 70, 2),
        restraints=True,
        force_energy_split=True,
        interspace=False,
        save=False,
        verbose=False
        ):
        """Given a umbrella sampling object, compute the free energy curves for number of hydrogen bonds, center-of-mass distance, and hb contacts order parameters.

        Args:
            us (MeltingUmbrellaSampling): _description_
            obs_df_subset (_type_, optional): _description_. Defaults to None.
            max_hb (int, optional): _description_. Defaults to 32.
            temp_range (_type_, optional): _description_. Defaults to np.arange(30, 70, 2).
        """        
        us_splits, u_knts = self._create_us_splits(convergence_splits, subsampling, temp_range, interspace=interspace)
        obs_df_whole = deepcopy(self.base_umbrella.obs_df)
        self.obs_df_whole = obs_df_whole
        
        free_energy_info = {'free_energies': {}, 'bins': {}, 'dfs': {}}
        for idx, (split, u_knt) in tqdm(enumerate(zip(us_splits, u_knts)), desc='Computing Free Energy Curves', total=len(us_splits)):
            # with suppress_output():
            frees, bins, dfs = self._compute_free_energy_curves_pymbar(
                op_strings=op_strings,
                obs_df_subset=split,
                n_bins=n_bins,
                temp_range=temp_range,
                restraints=restraints,
                force_energy_split=force_energy_split,
                obs_df_whole=obs_df_whole,
                u_knt=u_knt,
                verbose=verbose
                )

            free_energy_info['free_energies'][f'split_{idx}'] = frees
            free_energy_info['bins'][f'split_{idx}'] = bins
            free_energy_info['dfs'][f'split_{idx}'] = dfs


        # hbs: n_splits x 3 x n_temps x n_bins
        if (save is not False) and (save is not None):
            self._pymbar_folder_check()
            if save is True:
                save_number = len([file for file in os.listdir(self.pymbar_folder) if 'pymbar_convergence_data' in file])
                save_path = f'{self.pymbar_folder}/pymbar_convergence_data{save_number}.pkl'
            else:
                save_path = save
            with open(save_path, 'wb') as f:
                pickle.dump(free_energy_info, f)

        del self.obs_df_whole
        
        return free_energy_info


    def read_pymbar_convergence_data(self, save_path=None):
        if save_path is None:
            save_path = f'{os.path.abspath(self.base_umbrella.system_dir)}/pymbar/pymbar_convergence_data.pkl'
        
        with open(save_path, 'rb') as f:
            free_energy_info = pickle.load(f)
        # all_metrics = [hbs, contacts, coms]
        return free_energy_info
    
    
    def _compute_free_energy_curves_pymbar(
        self,
        op_strings=['hb_list', 'com_distance', 'hb_contact'],
        obs_df_subset=None,
        n_bins=50,
        temp_range=np.arange(30, 70, 2),
        restraints=True,
        force_energy_split=True,
        u_knt=None,
        obs_df_whole=None,
        verbose=False
        ):
        """Given a umbrella sampling object, compute the free energy curves for number of hydrogen bonds, center-of-mass distance, and hb contacts order parameters.

        Args:
            us (MeltingUmbrellaSampling): _description_
            obs_df_subset (_type_, optional): _description_. Defaults to None.
            max_hb (int, optional): _description_. Defaults to 32.
            temp_range (_type_, optional): _description_. Defaults to np.arange(30, 70, 2).
        """

        if obs_df_subset is not None:
            if obs_df_whole is None:
                obs_df_whole = [df.copy() for df in self.base_umbrella.obs_df]
            self.base_umbrella.obs_df = obs_df_subset

        try:
            self.run_mbar_fes(
                reread_files=False,
                sim_type='prod',
                uncorrelated_samples=False,
                restraints=restraints,
                force_energy_split=force_energy_split,
                verbose=verbose
                )
            
        except Exception as e:
            if obs_df_subset is not None:
                self.base_umbrella.obs_df = obs_df_whole
            raise e
            
        frees = {}
        bins = {}
        dfs = {}
        
        for op_string in op_strings:
            try:
                free, _bin, df = self.hist.fes_hist(op_string, n_bins=n_bins, uncorrelated_samples=False, temp_range=temp_range, u_knt=u_knt)
                frees[op_string] = free
                bins[op_string] = _bin
                dfs[op_string] = df       

            except Exception as e:
                print(f'Error in {op_string} histogram calculation: {traceback.format_exc()}\nContinuing with other order parameters and setting {op_string} values to np.nan.\n')
                frees[op_string] = np.nan
                bins[op_string] = np.nan
                dfs[op_string] = np.nan

        if obs_df_subset is not None:
            self.base_umbrella.obs_df = obs_df_whole

        return frees, bins, dfs
    
    
    def _pymbar_folder_check(self):
        pymbar_folder = f'{os.path.abspath(self.base_umbrella.system_dir)}/pymbar'
        if not os.path.exists(pymbar_folder):
            os.makedirs(pymbar_folder)
        self.pymbar_folder = pymbar_folder
    
    
    def _init_param_and_arrays(self):
        
        self.base_umbrella.info_utils.get_r0_values()
        self.base_umbrella.info_utils.get_stiff_value()
        self.base_umbrella.info_utils.get_temperature()
        self.base_umbrella.info_utils.get_n_particles_in_system()
        
        kB = 1 # Boltzmann constant
        temperature = self.base_umbrella.temperature # temperature
        beta = 1.0 / (kB * temperature) # inverse temperature of simulations

        K = len(self.base_umbrella.obs_df) # number of umbrellas
        T_k = np.ones(K, float) * temperature  # inital temperatures are all equal
        beta_k = np.ones(K, float) * beta # inverse temperatures of umbrellas

        min_length = min([len(inner_list['com_distance']) for inner_list in self.base_umbrella.obs_df])  
        max_length = max([len(inner_list['com_distance']) for inner_list in self.base_umbrella.obs_df])  

        N_max = max_length # maximum number of snapshots

        # Allocate storage for simulation data
        # N_k[k] is the number of snapshots from umbrella simulation k
        N_k = np.zeros([K], dtype=int)
        # K_k[k] is the spring constant (in sim_units) for umbrella simulation k
        K_k = np.zeros([K])
        # com0_k[k] is the spring center location (in sim_units) for umbrella simulation k
        com0_k = np.zeros([K])
        # com_kn[k,n] is the distance (in sim_units) for snapshot n from umbrella simulation k
        com_kn = np.zeros([K, N_max])
        # u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k
        u_kn = np.zeros([K, N_max])
        # u_res_kn[k,n] is the reduced potential energy of external resstraints (that are not the umbrella pot) at snapshot n of umbrella simulation k
        u_res_kn = np.zeros([K, N_max])
        #Statistical inefficency
        g_k = np.zeros([K])

        # u_kln[k,l,n] is the reduced potential energy of snapshot n from umbrella simulation k evaluated at umbrella l from umbrella restraints
        u_kln = np.zeros([K, K, N_max])
        
        return K, N_max, beta_k, N_k, K_k, com0_k, com_kn, u_kn, u_res_kn, u_kln


    def _pull_down_data(self, op_string, uncorrelated_samples=False):
        K = len(self.base_umbrella.obs_df)
        N_max = max([len(inner_list[op_string]) for inner_list in self.base_umbrella.obs_df])
        N_k = np.array([len(inner_list[op_string]) for inner_list in self.base_umbrella.obs_df])
        
        op_kn = [inner_list[op_string] for inner_list in self.base_umbrella.obs_df]
        op_kn = np.array([np.pad(inner_list, (0, N_max - len(inner_list)), 'constant') for inner_list in op_kn])
        op_n = pymbar.utils.kn_to_n(op_kn, N_k=N_k)
        
        if uncorrelated_samples is True:
            op_kn, N_k = self._subsample_correlated_data(K, N_k, op_kn)
            op_n = pymbar.utils.kn_to_n(op_kn, N_k=N_k)

        return op_kn, op_n, N_max, N_k
    
    
    def _subsample_correlated_data(self, K, N_k, op_kn):
        for k in range(K):
            t0, g, Neff_max = pymbar.timeseries.detect_equilibration_binary_search(op_kn[k, :])
            indices = pymbar.timeseries.subsample_correlated_data(op_kn[k, t0:], g=g)
            N_k[k] = len(indices)
            op_kn[k, 0 : N_k[k]] = op_kn[k, t0:][indices]
        return op_kn, N_k


    def _setup_input_data(self, reread_files=False, sim_type='prod', restraints=False, force_energy_split=False, uncorrelated_samples=False):
        if reread_files is False:
            if self.base_umbrella.obs_df == None:
                self.base_umbrella.analysis.read_all_observables(sim_type=sim_type)
        if reread_files is True:
            self.base_umbrella.analysis.read_all_observables(sim_type=sim_type)
        
        K, N_max, beta_k, N_k, K_k, com0_k, com_kn, u_kn, u_res_kn, u_kln = self._init_param_and_arrays()
        
        com_kn, op_n, N_max, N_k = self._pull_down_data('com_distance', uncorrelated_samples=uncorrelated_samples)

        if restraints is True:
            u_res_kn = self._setup_restrain_potential(com_kn, N_k, N_max, force_energy_split)
            u_res_kn = ((u_res_kn) * beta_k[:, np.newaxis])
        
        N = np.sum(N_k)
        
        com0_k = np.array(self.base_umbrella.r0) # umbrella potential centers        
        K_k = np.array([self.base_umbrella.stiff for _ in range(K)]) # umbrella potential stiffness
        
        for k in range(K):
            for n in range(int(N_k[k])):
                # Compute minimum-image torsion deviation from umbrella center l
                dchi = com_kn[k, n] - com0_k
                
                # Compute energy of snapshot n from simulation k in umbrella potential l
                if restraints is True:
                    u_kln[k, :, n] = beta_k[k] * (K_k / 2.0) * dchi**2 + u_res_kn[k, n] + u_kn[k, n]
                
                else:
                    u_kln[k, :, n] = beta_k[k] * (K_k / 2.0) * dchi**2 + u_kn[k, n]
                
        return u_kln, N_k
    
    
    def _setup_temp_scaled_potential(self, N_max, temp_range):
        
        names = ['backbone', 'bonded_excluded_volume', 'stacking', 'nonbonded_excluded_volume', 'hydrogen_bonding', 'cross_stacking', 'coaxial_stacking', 'debye_huckel']
        u_kn = [inner_list[names] for inner_list in self.base_umbrella.obs_df]
        
        # u_kn = np.array([np.pad(inner_list, ((0, N_max - len(inner_list)), (0,0)), 'constant') for inner_list in u_kn])
        kinetic_energy = [inner_list['kinetic_energy'] for inner_list in self.base_umbrella.obs_df]
        force_energy = [inner_list['force_energy_0'] for inner_list in self.base_umbrella.obs_df]
        non_pot_energy = [0 for kin,force in zip(kinetic_energy, force_energy)]
        
        # non_pot_energy = np.array([np.pad(inner_list, (0, N_max - len(inner_list)), 'constant') for inner_list in non_pot_energy])
        new_energy_per_window = self.base_umbrella._new_calcualte_bias_energy(non_pot_energy, temp_range, truncated_potential_energy=u_kn)
        new_energy_per_window = [val.swapaxes(1, 0) for val in new_energy_per_window]
        new_energy_per_window = np.array([np.pad(inner_list, ((0, N_max - len(inner_list)), (0,0)), 'constant') for inner_list in new_energy_per_window])
        new_energy_per_window = new_energy_per_window * self.base_umbrella.n_particles_in_system
        
        return new_energy_per_window


    def _setup_restrain_potential(self, com_kn, N_k, N_max, force_energy_split):
        if force_energy_split is True:
            number_of_forces = self.base_umbrella.info_utils.get_n_external_forces()
            n_restraints = number_of_forces - 2
            force_energy = [f'force_energy_{idx}' for idx in range(n_restraints)]
            u_res_kn = [np.sum(inner_list[force_energy], axis=1) for inner_list in self.base_umbrella.obs_df]
        else:
            n_restraints = 1
            force_energy = [f'force_energy_{idx}' for idx in range(n_restraints)]
            u_res_kn = [inner_list[force_energy[0]] for inner_list in self.base_umbrella.obs_df]
            umbrella_potentials_kn = [self.base_umbrella.w_i(com_n_k[:N_k_k], r0, self.base_umbrella.stiff)* 2 /self.base_umbrella.n_particles_in_system 
                                   for com_n_k, N_k_k, r0 in zip(com_kn, N_k, self.base_umbrella.r0)]
            u_res_kn = [res_pot - umb_pot for res_pot, umb_pot in zip(u_res_kn, umbrella_potentials_kn)]

        u_res_kn = np.array([np.pad(inner_list, (0, N_max - len(inner_list)), 'constant')
                       for inner_list in u_res_kn])
        u_res_kn = u_res_kn * self.base_umbrella.n_particles_in_system / 2
        
        return u_res_kn  


    def _create_us_splits(self, n_splits, subsample, temp_range, interspace=False):
        """
        Splits each window in the observation dataframe of the us object into n_splits,
        each subsampled by the specified subsample rate.

        Parameters:
        - us: An object with an 'obs_df' attribute containing a list of dataframes (windows).
        - n_splits: Number of splits to create for each window.
        - subsample: Subsampling rate.

        Returns:
        - A list of lists of dataframes, where each inner list corresponds to the splits of one window.
        """
        # Deep copy the observation dataframe windows
        # Initialize a list to store the splits for all windows
        splits = [[] for _ in range(n_splits)]
        
        self.base_umbrella.info_utils.get_temperature()
        self.base_umbrella.info_utils.get_n_particles_in_system()
        
        N_max = max([len(value['com_distance']) for value in self.base_umbrella.obs_df])
        u_knt = -self._setup_temp_scaled_potential(N_max, temp_range)
        
        u_knts = [[] for _ in range(n_splits)]
        for idx, window in enumerate(self.base_umbrella.obs_df):
            u_nt = u_knt[idx, :, :]
            if interspace is not False:
                window = window[::subsample].reset_index(drop=True)
                u_nt = u_nt[::subsample, :]

            # Create the remaining splits
            for i in range(n_splits):
                if interspace is False:
                    n_data = len(window)
                    start_index = (n_data // n_splits) * i
                    end_index = (n_data // n_splits) * (i + 1)
                    split = window[start_index:end_index:subsample].reset_index(drop=True)                    
                    splits[i].append(split)
                    
                    u_nts = u_nt[start_index:end_index:subsample, :]
                    u_knts[i].append(u_nts)
                else:
                    split = window[i::n_splits].reset_index(drop=True)
                    splits[i].append(split)
                    
                    u_nts = u_nt[i::n_splits, :]
                    u_knts[i].append(u_nts)
        
        N_maxes = [np.max([len(inner_list) for inner_list in new_energy_per_window]) for new_energy_per_window in u_knts]
        u_knts = [np.array([np.pad(inner_list, ((0, N_max - len(inner_list)), (0,0)), 'constant') for inner_list in new_energy_per_window]) for N_max,new_energy_per_window in zip(N_maxes, u_knts)]
        # all_data_subsampled = [window[::subsample] for window in windows]
        # splits.insert(0, all_data_subsampled)
        print(f'Data per window:\n {len(splits[0][0])}')

        return splits, u_knts


    def filter_nan_and_convert_to_array(self, values):
        return np.array([value for value in values if not np.all(np.isnan(value))])


    def calculate_melting_temperature(self, temp_range, probabilities):        
        #probabilities: n_temps x n_hb
        # prob = -log(G)
        bound_states = probabilities[:,1:].sum(axis=1)
        unbound_states = probabilities[:,0]
        ratio = bound_states / unbound_states 
    
        finf = 1. + 1. / (2. * ratio) - np.sqrt((1. + 1. / (2. * ratio))**2 - 1.)
        
        inverted_finfs = 1 - finf
        
        # Fit the sigmoid function to the inverted data
        p0 = [max(inverted_finfs), np.median(temp_range), 1, min(inverted_finfs)]  # initial guesses for L, x0, k, b
        try:
            popt, _ = curve_fit(self.base_umbrella.sigmoid, temp_range, inverted_finfs, p0)
        except:
            return np.nan, np.array([np.nan,]), np.array([np.nan,]), inverted_finfs
    
        # Generate fitted data
        x_fit = np.linspace(min(temp_range), max(temp_range), 500)
        y_fit = self.base_umbrella.sigmoid(x_fit, *popt)
        
        
        idx = np.argmin(np.abs(y_fit - 0.5))
        Tm = x_fit[idx]
        
        return Tm, x_fit, y_fit, inverted_finfs


    def plot_free_energy_with_error_bars(self, op_strings, free_energy_info, temp_range, p_value, legend=True, save_path=None):
        
        free_metrics = [
            self.filter_nan_and_convert_to_array(
                [free_metric[op_string] for free_metric in free_energy_info['free_energies'].values()]
            )
            for op_string in op_strings
        ]
        
        free_metrics = [metric - metric[:, :, 0][:, :, np.newaxis] for metric in free_metrics]
        mean_metrics = [np.nanmean(free_metric, axis=0) for free_metric in free_metrics]
        std_metrics = [np.nanstd(free_metric, axis=0) for free_metric in free_metrics]
        sem_metrics = [std_metric / np.sqrt(len(free_metric)) for free_metric, std_metric in zip(free_metrics, std_metrics)]

        df = free_metrics[0].shape[0] - 1
        t_score = t.ppf(1 - p_value/2, df)
        ci_metrics = [t_score * sem_metric for sem_metric in sem_metrics]

        bin_metrics = [
            self.filter_nan_and_convert_to_array(
                [free_metric[op_string] for free_metric in free_energy_info['bins'].values()]
            )[0]
        for op_string in op_strings
        ]

        x_labels = ['Number of H-bonds', 'HB Contacts / Max HBs', 'Center of Mass Distance (nm)'][::-1]
        degree = r'$^\circ$'
        fig, ax = plt.subplots(1,len(op_strings),dpi=300, figsize=(4 * 3 * 1.5, 3 * 2), sharey=True)
        for i in range(len(mean_metrics)):
            for j in range(len(mean_metrics[i])):
                ax[i].errorbar(bin_metrics[i], mean_metrics[i][j], yerr=ci_metrics[i][j], label=f'{temp_range[j]}{degree}C', fmt='.-', capsize=2.5, capthick=1, errorevery=1)
            ax[i].set_xlabel(op_strings[i])
        dg = r'$\Delta$G/$\mathregular{k_B}$T'
        fig.supylabel(f'Free Energy ({dg})')
        fig.tight_layout()
        
        if legend:
            ax[0].legend(loc='upper left')

        
        if save_path is not None:
            plt.savefig(save_path, transparent=True, dpi=300)
        
        metrics = {'mean': mean_metrics, 'std': std_metrics, 'sem': sem_metrics, 'ci': ci_metrics, 'bins': bin_metrics}
        return  metrics


    # def extrapolate_melting


    def plot_melting_curve(self, metric, max_hbs, temp_range, monomer_conc=None, magnesium_conc=None, ax=None, plot=True):

        if monomer_conc is not None:
            box_size = self.base_umbrella.molar_concentration_to_box_size(monomer_conc)
            xmax = max([max(innerlist['com_distance']) for innerlist in self.base_umbrella.obs_df])
            dg_v = self.base_umbrella.volume_correction(box_size, xmax)
            # dg_v = np.log((box_size**3) / ((4/3) * np.pi * xmax**3))

            f_i_temp = np.zeros_like(metric)
            f_i_temp[:,0] = metric[:,0] - dg_v
            f_i_temp[:,1:] = metric[:,1:]
            f_i_temp[:,:] = f_i_temp[:,:] + dg_v

            unnormed_prob = np.exp(-f_i_temp)
        else:
            unnormed_prob = np.exp(-metric)

        normed_prob = unnormed_prob / np.sum(unnormed_prob, axis=1)[: , np.newaxis]
        Tm, x_fit, y_fit, inverted_finfs = self.calculate_melting_temperature(temp_range, normed_prob)

        if plot is True:
            if ax is None:
                fig, ax = plt.subplots()

            ax.scatter(temp_range, inverted_finfs, marker='o')
            ax.plot(x_fit, y_fit, linestyle='--', linewidth=2)
            ax.axvline(x=Tm, color='r', linestyle='--', linewidth=2, label=f'Tm = {Tm:.2f} C')
            ax.legend()   

        if magnesium_conc is not None:
            na_tm = Tm
            fGC = 0.5
            Nbp = max_hbs
            mg_concentration = magnesium_conc
            mg_tm = self.base_umbrella.na_tm_to_mg_tm(na_tm, mg_concentration, fGC, Nbp)
            # print(f'MgTm: {mg_tm}')

            return Tm, inverted_finfs, mg_tm
        return Tm, inverted_finfs, np.nan
       
       
    def plot_melting_curve_with_error_bars(self, free_energy_info, op_string, n_tm_splits, max_hb, temp_range, p_val, monomer_conc=None, magnesium_conc=None, plot=True, ax=None, save_path=None):
        
        free_metric = self.filter_nan_and_convert_to_array(
                [free_metric[op_string] for free_metric in free_energy_info['free_energies'].values()]
            )
        
        free_metric = free_metric - free_metric[:, :, 0][:, :, np.newaxis] 

        def prime_factors(n):
            """Returns all the prime factors of a positive integer"""
            factors = []
            d = 2
            while n > 1:
                while n % d == 0:
                    factors.append(d)
                    n /= d
                d = d + 1
                if d*d > n:
                    if n > 1: factors.append(n)
                    break
            return factors
        n_tm_splits = max(prime_factors(len(free_metric)))

        np.random.shuffle(free_metric)

        free_metric_splits = np.array(np.array_split(free_metric, n_tm_splits, axis=0))
        free_metric_splits_mean = np.nanmean(free_metric_splits, axis=1) 
        tms = []
        inverted_infses = []
        mg_tms = []
        for metric in free_metric_splits_mean:
            Tm, inverted_finfs, mg_tm = self.plot_melting_curve(metric, max_hb, temp_range, monomer_conc=monomer_conc, magnesium_conc=magnesium_conc, ax=None, plot=False)
            tms.append(Tm)
            inverted_infses.append(inverted_finfs)
            mg_tms.append(mg_tm)

        mean_Tm = np.nanmean(tms)
        mean_inverted_finfs = np.nanmean(inverted_infses, axis=0)
        mean_mg_tm = np.nanmean(mg_tms)

        std_Tm = np.nanstd(tms)
        std_inverted_finfs = np.nanstd(inverted_infses, axis=0)
        std_mg_tm = np.nanstd(mg_tms)

        sem_Tm = std_Tm / np.sqrt(len(tms))
        sem_inverted_finfs = std_inverted_finfs / np.sqrt(len(inverted_infses))
        sem_mg_tm = std_mg_tm / np.sqrt(len(mg_tms))

        df = len(tms) - 1
        t_score = t.ppf(1 - p_val/2, df)
        ci_tm = t_score * sem_Tm
        ci_inverted_finfs = t_score * sem_inverted_finfs
        ci_mg_tm = t_score * sem_mg_tm

        if plot is True:
            p0 = [max(mean_inverted_finfs), np.median(temp_range), 1, min(mean_inverted_finfs)]  # initial guesses for L, x0, k, b

            popt, _ = curve_fit(self.base_umbrella.sigmoid, temp_range, mean_inverted_finfs, p0)


            x_fit = np.linspace(min(temp_range), max(temp_range), 500)
            y_fit = self.base_umbrella.sigmoid(x_fit, *popt)

            plus_minus = r'$\pm$'
            degree = r'$^\circ$'
            tm_string = r'$\mathregular{T_m}$'
            data_label = f'{tm_string} = {mean_Tm:.2f} {plus_minus} {ci_tm:.2f}{degree}C'
            legend_title = f'1 M Na+ simulated:\n{data_label}\n'
            
            if ax is None:
                fig, ax = plt.subplots(figsize=(4*2.25, 3*2.), dpi=300)
            ax.errorbar(temp_range, mean_inverted_finfs, yerr=ci_inverted_finfs, fmt='o', capsize=2.5, capthick=1, errorevery=1, label=legend_title)
            ax.plot(x_fit, y_fit, linestyle='--', linewidth=2, label=legend_title)
            ax.errorbar(mean_Tm, 0.5, xerr=ci_tm, fmt='o', capsize=2.5, capthick=1, errorevery=1)
            
            if magnesium_conc is not None:
                x_fit_mg = x_fit - (mean_Tm - mean_mg_tm)
                temp_range_mg = temp_range - (mean_Tm - mean_mg_tm) 
                x_add = np.linspace(min(temp_range) - (mean_Tm - mean_mg_tm), min(temp_range), 66)
                y_add = self.base_umbrella.sigmoid(x_add, *popt)
                x_fit_mg = x_fit_mg[x_fit_mg > np.min(x_fit)]
                y_fit_mg =  y_fit[-len(x_fit_mg):]
                
                x_fit_mg = np.concatenate((x_add, x_fit_mg))
                y_fit_mg = np.concatenate((y_add, y_fit_mg))
                
                mg_label = f'{tm_string} = {mean_mg_tm:.2f} {plus_minus} {ci_mg_tm:.2f}{degree}C'
                ax.errorbar(temp_range_mg, mean_inverted_finfs, yerr=ci_inverted_finfs, fmt='o', capsize=2.5, capthick=1, errorevery=1)
                ax.plot(x_fit_mg, y_fit_mg, linestyle='--', linewidth=2, label=f'20 mM Mg2+\n10 nM monomer')
                ax.errorbar(mean_mg_tm, 0.5, xerr=ci_mg_tm, fmt='o', capsize=2.5, capthick=1, errorevery=1, label=f'{mg_label}')


            ax.legend(loc='upper left') 
            
            ax.set_xlabel(f'Temperature ({degree}C)')
            ax.set_ylabel('Fraction of unbound monomer')

            fig.tight_layout()
            if save_path is not None:
                plt.savefig(save_path, transparent=True, dpi=300)
        return mean_Tm, ci_tm, mean_mg_tm, ci_mg_tm
        
    def plot_melting_across_monomer_concentration(self, all_metrics, n_tm_splits, max_hb, temp_range, p_val, monomer_conc_range, magnesium_conc=None, ax=None):
        
        mean_tms = []
        ci_tms = []
        for monomer_conc in monomer_conc_range:
            mean_Tm, ci_tm, mean_mg_tm, ci_mg_tm = self.plot_melting_curve_from_subsplits(all_metrics, n_tm_splits, max_hb, temp_range, p_val,
                                                                    monomer_conc=monomer_conc, magnesium_conc=magnesium_conc, plot=False)
        
            mean_tms.append(mean_mg_tm)
            ci_tms.append(ci_mg_tm)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(4*1.5, 3*2), dpi=300)
        ax.errorbar(np.log10(monomer_conc_range), mean_tms, yerr=ci_tms, fmt='-o', capsize=2.5, capthick=1, errorevery=1, label=f'{magnesium_conc:.2e}M Mg2+')
        log10 = r'$\mathregular{log_{10}}$'
        ax.set_xlabel(f'Monomer Concentration ({log10}M)')
        degree = r'$^\circ$'
        ax.set_ylabel(f'Melting Temperature ({degree}C)')
        ax.legend()
        

    def plot_melting_across_salt_concentration(self, all_metrics, n_tm_splits, max_hb, temp_range, p_val, magnesium_conc_range, monomer_conc=None, ax=None):
        
        mean_tms = []
        ci_tms = []
        for magnesium_conc in magnesium_conc_range:
            mean_Tm, ci_tm, mean_mg_tm, ci_mg_tm = self.plot_melting_curve_from_subsplits(all_metrics, n_tm_splits, max_hb, temp_range, p_val,
                                                                    monomer_conc=monomer_conc, magnesium_conc=magnesium_conc, plot=False)
        
            mean_tms.append(mean_mg_tm)
            ci_tms.append(ci_mg_tm)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(4*1.5, 3*1.5), dpi=300)
        ax.errorbar(np.log10(magnesium_conc_range), mean_tms, yerr=ci_tms, fmt='-o', capsize=2.5, capthick=1, errorevery=1, label=f'{monomer_conc:.1e}M DNA Monomer')
        log10 = r'$\mathregular{log_{10}}$'
        ax.set_xlabel(f'Magnesium Ion Concentration ({log10}M)')
        degree = r'$^\circ$'
        ax.set_ylabel(f'Melting Temperature ({degree}C)')
        ax.legend()
    
    
def _process_temp_idx(instance, temp_idx, u_kn, op_n, bin_edges, bin_center_i):
    results = instance._fes_histogram(u_kn, op_n, bin_edges, bin_center_i)
    f_i = results["f_i"]
    df_i = results["df_i"]
    f_i = f_i - f_i[0]
    return temp_idx, f_i, df_i
    
    
class PymbarHistograming:
    def __init__(self, pymbar_analysis):
        self.pymbar = pymbar_analysis
        self.base_umbrella = pymbar_analysis.base_umbrella
        
    def fes_hist(self, op_string, n_bins=50, uncorrelated_samples=False, temp_range=None, u_knt=None):
        K = len(self.base_umbrella.obs_df)
        op_kn, op_n, N_max, N_k = self.pymbar._pull_down_data(op_string, uncorrelated_samples=uncorrelated_samples)
        u_kn = np.zeros([K, N_max])
 
        bin_center_i, bin_edges, n_bins = self._choose_binning(op_string, n_bins)
        
        if temp_range is None:
            results = self._fes_histogram(u_kn, op_n, bin_edges, bin_center_i)
            center_f_i = results["f_i"]
            center_df_i = results["df_i"]
            center_f_i = center_f_i - center_f_i.min()
            bin_center_i = self._modify_bin_center(bin_center_i, op_string)
            return center_f_i, bin_center_i, center_df_i
        
        elif temp_range is not None:
            f_i = np.zeros([len(temp_range), len(bin_center_i)])
            df_i = np.zeros([len(temp_range), len(bin_center_i)])
            
            if u_knt is None:
                u_knt = -self.pymbar._setup_temp_scaled_potential(N_max, temp_range)
                
            # u_knt_iterable = (u_knt[:, :, temp_idx] for temp_idx in range(u_knt.shape[2]))
            # with ProcessPoolExecutor() as executor:
            #     results = list(executor.map(_process_temp_idx, [self] * len(temp_range) , temp_range, u_knt_iterable, [op_n] * len(temp_range), [bin_edges] * len(temp_range), [bin_center_i] * len(temp_range)))
            # for temp_idx, f_i_temp, df_i_temp in results:
            #     f_i[temp_idx, :] = f_i_temp
            #     df_i[temp_idx, :] = df_i_temp
            
            for temp_idx in range(u_knt.shape[2]):
                u_kn = u_knt[:, :, temp_idx]
                u_n = pymbar.mbar.kn_to_n(u_kn, N_k=N_k)
                results = self._fes_histogram(u_n, op_n, bin_edges, bin_center_i)
                f_i[temp_idx, :] = results["f_i"]
                df_i[temp_idx, :] = results["df_i"]
                f_i[temp_idx, :] = f_i[temp_idx, :] - f_i[temp_idx, 0]
                
            bin_center_i = self._modify_bin_center(bin_center_i, op_string)
            return f_i, bin_center_i, df_i
  

  
    def fes_2d_hist(self, op_string_0, op_string_1, n_bins=50, uncorrelated_samples=False, temp_range=None, u_knt=None):
        K = len(self.base_umbrella.obs_df)
        op_kn_0, op_n_0, N_max, N_k = self.pymbar._pull_down_data(op_string_0, uncorrelated_samples=uncorrelated_samples)
        op_kn_1, op_n_1, N_max, N_k = self.pymbar._pull_down_data(op_string_1, uncorrelated_samples=uncorrelated_samples)

        u_kn = np.zeros([K, N_max])
                
        bin_center_0, bin_edges_0, n_bins_0 = self._choose_binning(op_string_0, n_bins)
        bin_center_1, bin_edges_1, n_bins_1 = self._choose_binning(op_string_1, n_bins)     
        bin_edges = np.array([bin_edges_0, bin_edges_1])
        
        x_n = np.zeros([np.sum(N_k), 2])
        Ntot = 0
        for k in range(K):
            for n in range(N_k[k]):
                x_n[Ntot, 0] = op_kn_0[k, n]
                x_n[Ntot, 1] = op_kn_1[k, n]
                Ntot += 1
                
        bin_center_i, all_bin_centers = self._2d_binning(K, N_max, N_k, bin_edges_0, bin_edges_1, op_kn_0, op_kn_1, n_bins_0, n_bins_1)
        
        if temp_range is None:
            results = self._fes_histogram(u_kn, x_n, bin_edges, bin_center_i)
            f_i = results["f_i"]
            df_i = results["df_i"]
            f_i = f_i - f_i.min()
            
        elif temp_range is not None:
            f_i = np.zeros([len(temp_range), len(bin_center_i)])
            df_i = np.zeros([len(temp_range), len(bin_center_i)])
            
            if u_knt is None:
                u_knt = -self.pymbar._setup_temp_scaled_potential(N_max, temp_range)

            for temp_idx in range(u_knt.shape[2]):
                u_kn = u_knt[:, :, temp_idx]
                # u_n = pymbar.mbar.kn_to_n(u_kn, N_k=N_k)
                results = self._fes_histogram(u_kn, x_n, bin_edges, bin_center_i)
                f_i[temp_idx, :] = results["f_i"]
                df_i[temp_idx, :] = results["df_i"]
                f_i[temp_idx, :] = f_i[temp_idx, :] - f_i[temp_idx, 0]
                
            
        bin_centers = np.array(bin_center_i).T
        bin_centers[0,:] = self._modify_bin_center(bin_centers[0,:], op_string_0)
        bin_centers[1,:] = self._modify_bin_center(bin_centers[1,:], op_string_1)
        bin_centers = bin_centers.T
        f_i_bin_pair = {}
        for vals in range(len(f_i)):
            f_i_bin_pair[tuple(bin_centers[vals])] = f_i[vals]
        
        all_centers = []
        all_centers.append(sorted(self._modify_bin_center(np.array(list(all_bin_centers[0])), op_string_0)))
        all_centers.append(sorted(self._modify_bin_center(np.array(list(all_bin_centers[1])), op_string_1)))
        all_centers = np.array(all_centers)
        
        
        return f_i_bin_pair, all_centers, df_i
    
    
    def _fes_histogram(self, u_kn, op_n, bin_edges, bin_center_i):
        histogram_parameters = {}
        histogram_parameters["bin_edges"] = bin_edges
        self.pymbar.basefes.generate_fes(u_kn, op_n, fes_type="histogram", histogram_parameters=histogram_parameters)
        results = self.pymbar.basefes.get_fes(bin_center_i, reference_point="from-lowest", uncertainty_method="analytical")
        return results
    
    
    def _bin_centers(self, op_min, op_max, nbins, discrete=False):
        # compute bin centers
        bin_center_i = np.zeros([nbins])
        if discrete:
            bin_edges = np.arange(0, nbins + 1)
        else:
            bin_edges = np.linspace(op_min, op_max, nbins + 1)
        for i in range(nbins):
            bin_center_i[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])
            
        return bin_center_i, bin_edges
     
     
    def _choose_binning(self, op_string, n_bins):
        
        if hasattr(self, 'obs_df_whole') is True:
            df_to_use = self.pymbar.obs_df_whole
        else:
            df_to_use = self.base_umbrella.obs_df
        
        if op_string == 'hb_list':
            max_hb = int(np.max([np.max(inner_list[op_string]) for inner_list in df_to_use]))
            op_min = 0 # min of reaction coordinate
            op_max = max_hb # max of reaction coordinate
            n_bins = max_hb + 1 # number of bins
            bin_center_i, bin_edges = self._bin_centers(op_min, op_max, n_bins, discrete=True)
            
        elif op_string == 'com_distance':
            op_min = np.min([np.min(inner_list[op_string]) for inner_list in df_to_use]) + 1e-2 # min of reaction coordinate
            op_max = np.max([np.max(inner_list[op_string]) for inner_list in df_to_use]) -1e-1  # max of reaction coordinate

            # compute bin centers
            bin_center_i, bin_edges = self._bin_centers(op_min, op_max, n_bins)
            
        elif op_string == 'hb_contact':
            op_min = 0 # min of reaction coordinate
            op_max = 1 # max of reaction coordinate
            # compute bin centers
            bin_center_i, bin_edges = self._bin_centers(op_min, op_max, n_bins)
        
        else:
            op_min = np.min([np.min(inner_list[op_string]) for inner_list in df_to_use])
            op_max = np.max([np.max(inner_list[op_string]) for inner_list in df_to_use])
            bin_center_i, bin_edges = self._bin_centers(op_min, op_max, n_bins)
        
        return bin_center_i, bin_edges, n_bins
     
     
    def _modify_bin_center(self, bin_center_i, op_string):
        if op_string == 'hb_list':
            bin_center_i = bin_center_i - 0.5
        elif op_string == 'com_distance':
            bin_center_i = bin_center_i * 0.8518
        elif op_string == 'hb_contact':
            pass
        else:
            pass
        return bin_center_i
   

    def _2d_get_formatted_data(self, all_bin_centers, f_i_bin_pair):
        centers = np.full((len(all_bin_centers[0,:]), len(all_bin_centers[1,:])), np.nan)
        for bin_pair, f_i_val in f_i_bin_pair.items():
            for hb_idx, hbs in enumerate(all_bin_centers[0,:]):
                for com_idx, coms in enumerate(all_bin_centers[1,:]):
                    curr_pair = (hbs, coms)

                    if bin_pair == curr_pair:
                        centers[hb_idx, com_idx] = f_i_val
                        
        df = pd.DataFrame(centers, columns=all_bin_centers[1,:], index=all_bin_centers[0,:])
        df.dropna(axis=1, how='all')
        
        return df
  
    def _2d_binning(self, K, N_max, N_k, bin_edges_0, bin_edges_1, op_kn_0, op_kn_1, n_bins_0, n_bins_1):
        print("Binning...")


        # Create a list of indices of all configurations in kn-indexing.
        mask_kn = np.zeros([K, N_max], dtype=bool)
        for k in range(K):
            mask_kn[k, 0 : N_k[k]] = True
        # Create a list from this mask.
        indices = np.where(mask_kn)

        # Determine torsion bin size (in degrees)
        op_0_min = bin_edges_0[0]
        op_0_max = bin_edges_0[-1]

        op_1_min = bin_edges_1[0]
        op_1_max = bin_edges_1[-1]

        dx_0 = bin_edges_0[1] - bin_edges_0[0]
        dx_1 = bin_edges_1[1] - bin_edges_1[0]

        # Assign torsion bins
        # bin_kn[k,n] is the index of which histogram bin sample n from temperature index k belongs to
        bin_kn = np.zeros([K, N_max], dtype=int)

        bin_kn.shape
        nbins = 0
        bin_nonzero = 0
        bin_counts = []
        bin_centers = []  # bin_centers[i] is a (phi,psi) tuple that gives the center of bin i
        count_nonzero = []
        centers_nonzero = []
        all_bin_centers = [set(), set()]

        for i in range(n_bins_1):
            for j in range(n_bins_1):
                # Determine (phi,psi) of bin center.
                op_0 = op_0_min + dx_0 * (i + 0.5)
                op_1 = op_1_min + dx_1 * (j + 0.5)

                # Determine which configurations lie in this bin.
                in_bin = (
                    (op_0 - dx_0 / 2 <= op_kn_0[indices])
                    & (op_kn_0[indices] < op_0 + dx_0 / 2)
                    & (op_1 - dx_1 / 2 <= op_kn_1[indices])
                    & (op_kn_1[indices] < op_1 + dx_1 / 2)
                )
                # Count number of configurations in this bin.
                bin_count = in_bin.sum()
                # Generate list of indices in bin.
                # set bin indices of both dimensions
                if bin_count > 0:
                    count_nonzero.append(bin_count)
                    centers_nonzero.append((op_0, op_1))
                    bin_nonzero += 1
                    
                all_bin_centers[0].add(op_0)
                all_bin_centers[1].add(op_1)
        
        all_bin_centers
        
        return centers_nonzero, all_bin_centers
