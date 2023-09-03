import subprocess
from oxdna_simulation import Simulation
import os
import matplotlib.pyplot as plt
import shutil
from os.path import abspath, join, exists
import multiprocessing as mp
import pandas as pd
import numpy as np
import pickle as pkl


class Metadynamics:
    def __init__(self, file_dir, sim_dir):
        self.sim = Simulation(file_dir, sim_dir)
        self.meta_build = MetadynamicsBuild(self)
        self.meta_run = MetadynamicsRun(self)
        self.meta_analysis = MetadynamicsAnalysis(self)
        
    def build(self, op_1, op_2):
        self.meta_build.prepare_run(op_1, op_2)
        
    def run(self, meta_arguments):
        self.meta_run.run(meta_arguments)
        

class MetadynamicsBuild:
    def __init__(self, meta):
        self.meta = meta
        
    def prepare_run(self, set_1, set_2):
        self.meta.sim.build(clean_build='force')
        self.meta.sim.input_file({'external_forces_as_JSON':'false'})
        self.write_locs(set_1, set_2)
        self.create_base()
        
    def write_locs(self, set_1, set_2):
        with open(abspath(f'{self.meta.sim.sim_dir}/locs.meta'), 'w') as f:
            f.write(f'p1a:{set_1}')
            f.write('\n')
            f.write(f'p2a:{set_2}')
    
    def create_base(self):
        if not exists(join(self.meta.sim.sim_dir, 'base')):
            os.mkdir(join(self.meta.sim.sim_dir, 'base'))
        try:
            shutil.copyfile(self.meta.sim.sim_files.dat, join(self.meta.sim.sim_dir, 'base', self.meta.sim.sim_files.dat.split('/')[-1]))
        except:
            shutil.copyfile(self.meta.sim.sim_files.last_conf, join(self.meta.sim.sim_dir, 'base', self.meta.sim.sim_files.last_conf.split('/')[-1]))
        
        shutil.copyfile(self.meta.sim.sim_files.top, join(self.meta.sim.sim_dir, 'base', self.meta.sim.sim_files.top.split('/')[-1]))
        shutil.copyfile(self.meta.sim.sim_files.input, join(self.meta.sim.sim_dir, 'base', self.meta.sim.sim_files.input.split('/')[-1]))
        

class MetadynamicsRun:
    def __init__(self, meta):
        self.meta = meta
        
    def run_metadynamics(self, meta_arguments):
        self.meta_arguments = meta_arguments
        self.meta_arguments = {meta_arguments[0 + i]:meta_arguments[1 + i] for i in range(int( len(meta_arguments) / 2))}    
        os.chdir(self.meta.sim.sim_dir)
        completed_process = subprocess.run(["python3",
                                            abspath("/scratch/mlsample/ipy_oxDNA/src/metad_interface.py"),
                                            join(abspath(self.meta.sim.sim_dir), "base"),
                                            *meta_arguments]
                                           , capture_output=True, check=True)
        return completed_process
    
    def run(self, meta_arguments, log=None, join=False, gpu_mem_block=True, custom_observables=None, run_when_failed=False):
        """ Run the worker manager in a subprocess"""
        print('spawning')
        self.meta_arguments = meta_arguments
        self.meta_arguments = {meta_arguments[0 + i]:meta_arguments[1 + i] for i in range(int( len(meta_arguments) / 2))}
        p = mp.Process(target=self.run_metadynamics, args=(meta_arguments, )) 
        self.manager_process = p
        p.start()
        if join == True:
            p.join()     

            
class MetadynamicsAnalysis:
    def __init__(self, meta):
        self.meta = meta
    
    def load_all_positions(self, index):
        data = pd.read_csv(f'{self.meta.sim.sim_dir}/run-meta_{index}/pos.dat',header=None)
        return np.array(data).flatten()
    
    def plot_distance(self, n_walkers):
        positions = [self.load_all_positions(i) for i in range(n_walkers)]
        for idx,pos in enumerate(positions):
            x = np.array(pos).flatten()[::10] * 0.85
            plt.plot(np.arange(len(x)),x, label=f'Walker {idx}')
        plt.xlabel('timesteps (x10,000)')
        plt.ylabel('com_distance')
        plt.legend(fontsize=6)
        
    def load_bias(self, fname):
        return pkl.load(open(f"{self.meta.sim.sim_dir}/bias/{fname}",'rb'))
        
    def plot_free_energy(self):
        bias_files = os.listdir(f'{self.meta.sim.sim_dir}/bias/')
        bias_dictionary = {int(x.split('_')[1]) : x for x in bias_files}
        keys = np.sort(list(bias_dictionary.keys()))
        # biases = [load_bias(bias_dictionary[key]) for key in keys]
        biases = [self.load_bias(bias_dictionary[keys[-1]])]
        to_plot = -(biases[-1] - np.max(biases[-1]))
        plt.plot(np.arange(0,80.001,0.001)*0.85, to_plot)

        
    def plot_convergence(self):
            
        bias_files = os.listdir(f'{self.meta.sim.sim_dir}/bias/')
        bias_dictionary = {int(x.split('_')[1]) : x for x in bias_files}
        keys = np.sort(list(bias_dictionary.keys()))
        biases = [self.load_bias(bias_dictionary[key]) for key in keys]
        
        # the centers stores the metadynamic bias as a function of time
        # in well-tempered metadynamics, the bias converges to a multiple of the free energy
        f,ax = plt.subplots(1,1,sharex = True,sharey=True)
        from matplotlib.cm import rainbow
        for index,i in enumerate(biases[1::10]):
            c = i
            c = -c*21/20 
            # 21/20 because metadynamic Delta T is equal to 20T, and the relationship between converged metadynamic bias and free energy is:
            # free energy = (T + Delta T) / T
            # see, for example, Bussi et al. 2020
            c -= np.min(c) 
            # minimum of free energy is arbitrary
            plt.plot(np.arange(0,80.001,0.001)*0.85,c,color = rainbow(index/len(biases) * 10))
            # 0.85 is to convert from oxDNA units to nm
            # range of coordiantes is from the grid spacing, see these values in the run.sh script.
        
        plt.xlabel('E2E distance (nm)')
        plt.ylabel('free energy (k$_B$T)')