import os
import numpy as np
import shutil
from json import dumps, loads
import oxpy
import multiprocessing as mp
import py
from oxDNA_analysis_tools.UTILS.oxview import oxdna_conf
from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs
import ipywidgets as widgets
from IPython.display import display, IFrame
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import nvidia_smi


class Simulation:
    def __init__(self, file_dir, sim_dir, exsisting=True):
        self.file_dir = file_dir
        self.sim_dir = sim_dir
        self.sim_files = SimFiles(self.sim_dir)
        self.build_sim = BuildSimulation(self)
        self.input = Input(self.sim_dir)
        self.analysis = Analysis(self)
        self.oxpy_run = OxpyRun(self)
    
    def build(self, clean_build=False):
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
        self.sim_files.parse_current_files()
    
    def input_file(self, parameters):
        self.input.modify_input(parameters)
    
    def add_force(self, force_js):
        if not os.path.exists(os.path.join(self.sim_dir, "forces.json")):
            self.input_file({'external_forces':'1'})
        self.build_sim.build_force(force_js)
        
    def add_observable(self, observable_js):
        if not os.path.exists(os.path.join(self.sim_dir, "observables.json")):
            self.input_file({'observables_file': 'observables.json'})
        self.build_sim.build_observable(observable_js)

    def slurm_run(self, run_file, job_name='oxDNA'):
        self.sim_files.run_file = os.path.abspath(os.path.join(self.sim_dir, run_file))
        self.slurm_run = SlurmRun(self.sim_dir, run_file, job_name)
    
    def sequence_dependant(self):
        self.input_file({'use_average_seq': 'no', 'seq_dep_file':'oxDNA2_sequence_dependent_parameters.txt'})
        SequenceDependant(self.sim_dir)
          

    
class BuildSimulation:
    """Methods used to create/build oxDNA simulation"""
    def __init__(self, sim):
        self.sim = sim
        self.file_dir = sim.file_dir
        self.sim_dir = sim.sim_dir
        self.force = Force()
    
    def get_last_conf_top(self):
        """set attributes containing the name of the inital conf (dat file) and topology"""
        conf_top = os.listdir(self.file_dir)
        self.top = [file for file in conf_top if (file.endswith(('.top')))][0]
        try:
            last_conf = [file for file in conf_top if (file.startswith(('last_conf')))][0]
        except IndexError:
            last_conf = [file for file in conf_top if (file.endswith(('.dat'))) and not (file.endswith(('energy.dat'))) and not (file.endswith(('trajectory.dat'))) and not (file.endswith(('error_conf.dat')))][0]
        self.dat = last_conf
        
    def build_sim_dir(self):
        """Make the simulation directory"""
        if not os.path.exists(self.sim_dir):
            os.mkdir(self.sim_dir)
            
    def build_dat_top(self):
        """Write intial conf and toplogy to simulation directory"""
        self.get_last_conf_top()
        shutil.copy(os.path.join(self.file_dir, self.dat), self.sim_dir)
        shutil.copy(os.path.join(self.file_dir, self.top), self.sim_dir)
          
    def build_input(self, production=False):
        """Calls a methods from the Input class which writes a oxDNA input file in plain text and json"""
        self.sim.input = Input(self.sim_dir)
        self.sim.input.write_input(production=production)  
    
    def build_force(self, force_js):
        """
        Write force file is one does not exist. If a force file exists add additional forces to the file.
        
        Parameters:
            force_js (dict): force dictornary obtained from the Force class methods
        """
        if not os.path.exists(os.path.join(self.sim_dir, "forces.json")):
            with open(os.path.join(self.sim_dir, "forces.json"), 'w') as f:
                f.write(dumps(force_js, indent=4))
        else:
            with open(os.path.join(self.sim_dir, "forces.json"), 'r') as f:
                read_force_js = loads(f.read())
                for force in list(read_force_js.values()):
                    if list(force.values())[1] == list(list(force_js.values())[0].values())[1]:
                        return None
                read_force_js[f'force_{len(list(read_force_js.keys()))}'] = read_force_js['force']
                del read_force_js['force']
                read_force_js.update(force_js.items())
                with open(os.path.join(self.sim_dir, "forces.json"), 'w') as f:
                    f.write(dumps(read_force_js, indent=4))
    
    def build_observable(self, observable_js):
        """
        Write observable file is one does not exist. If a observable file exists add additional observables to the file.
        
        Parameters:
            observable_js (dict): observable dictornary obtained from the Observable class methods
        """
        if not os.path.exists(os.path.join(self.sim_dir, "observables.json")):
            with open(os.path.join(self.sim_dir, "observables.json"), 'w') as f:
                f.write(dumps(observable_js, indent=4))
        else:
            with open(os.path.join(self.sim_dir, "observables.json"), 'r') as f:
                read_observable_js = loads(f.read())
                for observable in list(read_observable_js.values()):
                    if list(observable.values())[1] == list(list(observable_js.values())[0].values())[1]:
                        return None
                read_observable_js[f'output_{len(list(read_observable_js.keys()))}'] = read_observable_js['output']
                del read_observable_js['output']
                read_observable_js.update(observable_js.items())
                with open(os.path.join(self.sim_dir, "observables.json"), 'w') as f:
                    f.write(dumps(read_observable_js, indent=4))    

class OxpyRun:
    """Automatically runs a built oxDNA simulation using oxpy within a subprocess. Runs complete unless a number of steps is specified"""
    def __init__(self, sim):
        self.sim = sim
        self.sim_dir = sim.sim_dir
            
    def run(self, subprocess=True, steps=None, continue_run=False, verbose=True, log=True, join=False):
        self.manager = mp.Manager()
        self.sim_output = self.manager.Namespace()
        self.subprocess = subprocess
        self.steps = steps
        self.verbose = verbose
        self.continue_run = continue_run
        self.log = log
        self.join = join
        if self.verbose == True:
            print(f'Running: {self.sim_dir}')
        if self.subprocess:
            if self.steps is None:
                self.spawn(self.run_complete)
            elif self.steps is not None:
                self.spawn(self.run_steps)
        else:
            if self.steps is None:
                self.run_complete()
            elif self.steps is not None:
                self.run_steps()
        
            
    def spawn(self, f, args=()):
        """Spawn subprocess"""
        p = mp.Process(target=f, args=args)
        p.start()
        if self.join == True:
            p.join()
        self.process = p
    
    def run_complete(self):
        """Run an oxDNA simulation to completion using the number of step defined in the input file"""
        capture = py.io.StdCaptureFD()
        if self.continue_run == True:
            self.sim.input_file({"conf_file": self.sim.sim_files.last_conf, "refresh_vel": "0", "restart_step_counter": "0"})
        os.chdir(self.sim_dir)
        with open(os.path.join(self.sim_dir, 'input.json'), 'r') as f:
            my_input = loads(f.read())
        with oxpy.Context():
            ox_input = oxpy.InputFile()
            for k, v in my_input.items():
                ox_input[k] = v
            manager = oxpy.OxpyManager(ox_input)
            manager.run_complete()
        self.sim_output.out = capture.reset()
        if self.verbose == True:
            print(f'Finished: {self.sim_dir}')
        if self.log == True:
            with open('log.log', 'w') as f:
                f.write(self.sim_output.out[0])
                f.write(self.sim_output.out[1])
        self.sim.sim_files.parse_current_files()
            
        
    def run_steps(self):
        """Run an oxDNA simulation for a user specificed number of steps"""
        capture = py.io.StdCaptureFD()
        if self.continue_run == True:
            self.sim.input_file({"conf_file": self.sim.sim_files.last_conf, "refresh_vel": "0", "restart_step_counter": "0"})
        os.chdir(self.sim_dir)
        with open(os.path.join(self.sim_dir, 'input.json'), 'r') as f:
            my_input = loads(f.read())
        with oxpy.Context():
            ox_input = oxpy.InputFile()
            for k, v in my_input.items():
                ox_input[k] = v
            manager = oxpy.OxpyManager(ox_input)
            manager.run(self.steps) 
        self.sim_output.out = capture.reset()
        if self.verbose == True:
            print(f'Finished: {self.sim_dir}')
        if self.log == True:
            with open('log.log', 'w') as f:
                f.write(self.sim_output.out[0])
                f.write(self.sim_output.out[1])
        self.sim.sim_files.parse_current_files()
        
        
class SlurmRun:
    """Using a user provided slurm run file, setup a slurm job to be run"""
    def __init__(self, sim_dir, run_file, job_name):
        self.sim_dir = sim_dir
        self.run_file = run_file
        self.job_name = job_name
        self.write_run_file()
    
    def write_run_file(self):
        
        with open(self.run_file, 'r') as f:
            lines = f.readlines()
            with open(os.path.join(self.sim_dir, 'run.sh'), 'w') as r:
                for line in lines:
                    if 'job-name' in line:
                        r.write(f'#SBATCH --job-name="{self.job_name}"\n')
                    else:
                        r.write(line)
    def sbatch(self):
        os.chdir(self.sim_dir)
        os.system("sbatch run.sh")             


class SimulationManager:
    def __init__(self, n_processes=len(os.sched_getaffinity(0))-2):
        self.n_processes = n_processes
        self.manager = mp.Manager()
        self.sim_queue = self.manager.Queue()
        self.process_queue = self.manager.Queue(self.n_processes)
        self.gpu_memory_queue = self.manager.Queue(1)
  
    def gpu_resources(self):
        avalible_memory = []
        nvidia_smi.nvmlInit()
        NUMBER_OF_GPU = nvidia_smi.nvmlDeviceGetCount()
        for i in range(NUMBER_OF_GPU):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            avalible_memory.append(self._bytes_to_megabytes(info.total) - self._bytes_to_megabytes(info.used))
            gpu_most_aval_mem_free = max(avalible_memory)
            gpu_most_aval_mem_free_idx = avalible_memory.index(gpu_most_aval_mem_free)
        return np.round(gpu_most_aval_mem_free, 2), gpu_most_aval_mem_free_idx

    def _bytes_to_megabytes(self, bytes):
        return round((bytes/1024)/1024,2)

    def get_sim_mem(self, sim, gpu_idx):
        sim.input_file({'lastconf_file':'temp'})
        sim.oxpy_run.run(steps=0, subprocess=False, verbose=False, log=False)
        sim.input_file({'lastconf_file':'last_conf.dat'})
        mem_get = True
        err_split = sim.oxpy_run.sim_output.out[1].split()
        mem = err_split.index('memory:')
        sim_mem = err_split[mem + 1]
        #os.remove('./trajectory.dat')
        os.remove('./temp')
        #os.remove('./energy.dat')
        #os.remove('./error_conf.dat')
        return float(sim_mem)
    
    def queue_sim(self, sim, continue_run=False):
        if continue_run == True:
            sim.input_file({"conf_file": sim.sim_files.last_conf, "refresh_vel": "0", "restart_step_counter": "0"})
        self.sim_queue.put(sim)   
                    
    def worker_manager(self):
        while not self.sim_queue.empty():
            #get simulation from queue
            sim = self.sim_queue.get()
            free_gpu_memory, gpu_idx = self.gpu_resources()
            sim.input_file({'CUDA_device': str(gpu_idx)})
            p = mp.Process(target=self.worker_job, args=(sim, gpu_idx))
            p.start()
            sim_mem = self.gpu_memory_queue.get()
            self.process_queue.put('Simulation worker finished')
            if free_gpu_memory < (3 * sim_mem):
                wait_for_gpu_memory = True
                while wait_for_gpu_memory == True:
                    if free_gpu_memory < (3 * sim_mem):
                        free_gpu_memory, gpu_idx = self.gpu_resources()
                        sleep(5)
                    else:
                        print('gpu memory freed')
                        wait_for_gpu_memory = False      
        while not self.process_queue.empty():
            sleep(1)
        print('All queued simulations finished')
            
    def worker_job(self, sim, gpu_idx):
            sim_mem = self.get_sim_mem(sim, gpu_idx)
            self.gpu_memory_queue.put(sim_mem)
            sim.oxpy_run.run(subprocess=False)
            print(self.process_queue.get())
    
    def run(self, join=False):
        print('spawning')
        p = mp.Process(target=self.worker_manager, args=()) 
        p.start()
        if join == True:
            p.join()
    
    def start_nvidia_cuda_mps_control(self):
        os.system("""export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe_$SLURM_TASK_PID;
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log_$SLURM_TASK_PID;
mkdir -p $CUDA_MPS_PIPE_DIRECTORY;
mkdir -p $CUDA_MPS_LOG_DIRECTORY;
nvidia-cuda-mps-control -d;""")

        
                    
class Input:
    def __init__(self, sim_dir, parameters=None):
        self.sim_dir = sim_dir
        if os.path.exists(os.path.join(self.sim_dir, 'input.json')):
            self.read_input()
        else:
            self.input = {
            "interaction_type": "DNA2",
            "salt_concentration": "1.0",
            "sim_type": "MD",
            "backend": "CUDA",
            "backend_precision": "mixed",
            "use_edge": "1",
            "edge_n_forces": "1",
            "CUDA_list": "verlet",
            "CUDA_sort_every": "0",
            "max_density_multiplier": "2",
            "steps": "1e9",
            "ensemble": "nvt",
            "thermostat": "john",
            "T": "20C",
            "dt": "0.003",
            "verlet_skin": "0.5",
            "diff_coeff": "2.5",
            "newtonian_steps": "103",
            "topology": None,
            "conf_file": None,
            "lastconf_file": "last_conf.dat",
            "trajectory_file": "trajectory.dat",
            "refresh_vel": "1",
            "no_stdout_energy": "0",
            "restart_step_counter": "1",
            "energy_file": "energy.dat",
            "print_conf_interval": "1e5",
            "print_energy_every": "1e5",
            "time_scale": "linear",
            "max_io": "5",
            "external_forces": "0",
            "external_forces_file": "forces.json",
            "external_forces_as_JSON": "true"
            }
            if parameters != None:
                for k, v in parameters.items():
                    self.input[k] = v
    
    def get_last_conf_top(self):
        conf_top = os.listdir(self.sim_dir)
        self.top = [file for file in conf_top if (file.endswith(('.top')))][0]
        try:
            last_conf = [file for file in conf_top if (file.startswith(('last_conf')))][0]
        except IndexError:
            last_conf = [file for file in conf_top if (file.endswith(('.dat'))) and not (file.endswith(('energy.dat'))) and not (file.endswith(('trajectory.dat'))) and not (file.endswith(('error_conf.dat')))][0]
        self.dat = last_conf
        
    def write_input_standard(self):
        with oxpy.Context():
            ox_input = oxpy.InputFile()
            for k, v in self.input.items():
                ox_input[k] = v
            print(ox_input, file=f)
    
    def write_input(self, production=False):
        if production is False:
            self.get_last_conf_top()
            self.input["conf_file"] = self.dat
            self.input["topology"] = self.top
        #Write input file
        with open(os.path.join(self.sim_dir, f'input.json'), 'w') as f:
            input_json = dumps(self.input, indent=4)
            f.write(input_json)
        with open(os.path.join(self.sim_dir, f'input'), 'w') as f:
            with oxpy.Context(print_coda=False):
                ox_input = oxpy.InputFile()
                for k, v in self.input.items():
                    ox_input[k] = v
                print(ox_input, file=f)    
        
    def modify_input(self, parameters):
        for k, v in parameters.items():
                self.input[k] = v
        self.write_input()
                         
    def read_input(self):
        with open(os.path.join(self.sim_dir, 'input.json'), 'r') as f:
            my_input = loads(f.read())
        self.input = my_input

        
class SequenceDependant:
    def __init__(self, sim_dir):
        self.sim_dir = sim_dir
        self.parameters = """STCK_FACT_EPS = 0.18
STCK_G_C = 1.69339
STCK_C_G = 1.74669
STCK_G_G = 1.61295
STCK_C_C = 1.61295
STCK_G_A = 1.59887
STCK_T_C = 1.59887
STCK_A_G = 1.61898
STCK_C_T = 1.61898
STCK_T_G = 1.66322
STCK_C_A = 1.66322
STCK_G_T = 1.68032
STCK_A_C = 1.68032
STCK_A_T = 1.56166
STCK_T_A = 1.64311
STCK_A_A = 1.84642
STCK_T_T = 1.58952
HYDR_A_T = 0.88537
HYDR_T_A = 0.88537
HYDR_C_G = 1.23238
HYDR_G_C = 1.23238"""
        self.write_sequence_dependant_file()
    
    def write_sequence_dependant_file(self):
        with open(os.path.join(self.sim_dir,'oxDNA2_sequence_dependent_parameters.txt'), 'w') as f:
            f.write(self.parameters)
        
        
class Analysis:
    def __init__(self, simulation):
        self.sim_files = simulation.sim_files
        self.input = simulation.input.input
        
    def get_init_conf(self):
        self.sim_files.parse_current_files()
        ti, di = describe(self.sim_files.top,
                          self.sim_files.dat)
        return (ti, di), get_confs(ti, di, 0, 1)[0]
    
    def get_last_conf(self):
        self.sim_files.parse_current_files()
        ti, di = describe(self.sim_files.top,
                          self.sim_files.last_conf)
        return (ti,di), get_confs(ti, di, 0,1)[0]
    
    def view_init(self):
        (ti,di), conf = self.get_init_conf()        
        oxdna_conf(ti, conf)
                          
    def view_last(self):
        (ti,di), conf = self.get_last_conf()
        oxdna_conf(ti, conf)
    
    def get_conf_count(self):
        self.sim_files.parse_current_files()
        ti,di = describe(self.sim_files.top,
                         self.sim_files.traj)
        return len(di.idxs)
    
    def get_conf(self, id:int):
        self.sim_files.parse_current_files()
        ti,di = describe(self.sim_files.top,
                         self.sim_files.traj)
        l = len(di.idxs)
        if(id < l):
            return (ti,di), get_confs(ti,di, id, 1)[0]
        else:
            raise Exception("You requested a conf out of bounds.")
    
    def current_step(self):
        n_confs = float(self.get_conf_count())
        steps_per_conf = float(self.input["print_conf_interval"])
        return n_confs * steps_per_conf
    
    def view_conf(self, id:int):
        (ti,di), conf = self.get_conf(id)
        oxdna_conf(ti, conf)

    def plot_energy(self):
        self.sim_files.parse_current_files()
        df = pd.read_csv(self.sim_files.energy, delimiter="\s+",names=['time', 'U','P','K'])
        dt = float(self.input["dt"])
        steps = float(self.input["steps"])
        # make sure our figure is bigger
        plt.figure(figsize=(15,3)) 
        # plot the energy
        plt.plot(df.time/dt,df.U)
        plt.ylabel("Energy")
        plt.xlabel("Steps")
        # and the line indicating the complete run
        plt.ylim([-2,0])
        plt.plot([steps,steps],[0,-2], color="r")                
#Unstable
#     def view_traj(self,  init = 0, op=None):
#         # get the initial conf and the reference to the trajectory 
#         (ti,di), cur_conf = self.get_conf(init)
        
#         slider = widgets.IntSlider(
#             min = 0,
#             max = len(di.idxs),
#             step=1,
#             description="Select:",
#             value=init
#         )
        
#         output = widgets.Output()
#         if op:
#             min_v,max_v = np.min(op), np.max(op)
        
#         def handle(obj=None):
#             conf= get_confs(ti,di,slider.value,1)[0]
#             with output:
#                 output.clear_output()
#                 if op:
#                     # make sure our figure is bigger
#                     plt.figure(figsize=(15,3)) 
#                     plt.plot(op)
#                     print(init)
#                     plt.plot([slider.value,slider.value],[min_v, max_v], color="r")
#                     plt.show()
#                 oxdna_conf(ti,conf)
                
#         slider.observe(handle)
#         display(slider,output)
#         handle(None)


class Observable:
    @staticmethod
    def distance(particle_1=None, particle_2=None, print_every=None, name=None):
        return({
            "output": {
                "print_every": print_every,
                "name": name,
                "cols": [
                    {
                        "type": "distance",
                        "particle_1": particle_1,
                        "particle_2": particle_2
                    }
                ]
            }
        })
        
              
class Force:
    
    @staticmethod
    def com_force(com_list=None, ref_list=None, stiff=None, r0=None, PBC=None, rate=None):
        return({"force":{
                "type":'com',
                "com_list": f'{com_list}',
                "ref_list": f'{ref_list}',
                "stiff": f'{stiff}',
                "r0": f'{r0}',
                "PBC": f'{PBC}',
                "rate": f'{rate}'
                        }
        })
    
    @staticmethod
    def mutual_trap(particle=None, ref_particle=None, stiff=None, r0=None, PBC=None):
        """
        A spring force that pulls a particle towards the position of another particle
    
        Parameters:
            particle (int): the particle that the force acts upon
            ref_particle (int): the particle that the particle will be pulled towards
            stiff (float): the force constant of the spring (in simulation units)
            r0 (float): the equlibrium distance of the spring
            PBC (bool): does the force calculation take PBC into account (almost always 1)
        """
        return({
            "type" : "mutual_trap",
            "particle" : particle,
            "ref_particle" : ref_particle,
            "stiff" : stiff, 
            "r0" : r0,
            "PBC" : PBC
        })
    
        
    @staticmethod
    def string(particle, f0, rate, direction):
        """
        A linear force along a vector
    
        Parameters:
            particle (int): the particle that the force acts upon
            f0 (float): the initial strength of the force at t=0 (in simulation units)
            rate (float or SN string): growing rate of the force (simulation units/timestep)
            dir ([float, float, float]): the direction of the force
        """
        return({
            "type" : "string",
            "particle" : particle, 
            "f0" : f0, 
            "rate" : rate, 
            "dir" : direction 
        })
    
        
    @staticmethod
    def harmonic_trap(particle, pos0, stiff, rate, direction):
        """
        A linear potential well that traps a particle
    
        Parameters:
            particle (int): the particle that the force acts upon
            pos0 ([float, float, float]): the position of the trap at t=0
            stiff (float): the stiffness of the trap (force = stiff * dx)
            rate (float): the velocity of the trap (simulation units/time step)
            direction ([float, float, float]): the direction of movement of the trap
        """
        return({
            "type" : "trap",
            "particle" : particle, 
            "pos0" : pos0,
            "rate" : rate,
            "dir" : direction
        })
    
        
    @staticmethod
    def rotating_harmonic_trap(particle, stiff, rate, base, pos0, center, axis, mask):
        """
        A harmonic trap that rotates in space with constant angular velocity
    
        Parameters:
            particle (int): the particle that the force acts upon
            pos0 ([float, float, float]): the position of the trap at t=0
            stiff (float): the stiffness of the trap (force = stiff * dx)
            rate (float): the angular velocity of the trap (simulation units/time step)
            base (float): initial phase of the trap
            axis ([float, float, float]): the rotation axis of the trap
            mask([float, float, float]): the masking vector of the trap (force vector is element-wise multiplied by mask)
        """
        return({
            "type" : "twist", 
            "particle" : particle,
            "stiff" : stiff,
            "rate" : rate,
            "base" : base,
            "pos0" : pos0,
            "center" : center,
            "axis" : axis,
            "mask" : mask
        })
    
        
    @staticmethod
    def repulsion_plane(particle, stiff, direction, position):
        """
        A plane that forces the affected particle to stay on one side.
    
        Parameters:
            particle (int): the particle that the force acts upon.  -1 will act on whole system.
            stiff (float): the stiffness of the trap (force = stiff * distance below plane)
            dir ([float, float, float]): the normal vecor to the plane
            position(float): position of the plane (plane is d0*x + d1*y + d2*z + position = 0)
        """
        return({
            "type" : "repulsion_plane",
            "particle" : particle,
            "stiff" : stiff,
            "dir" : direction,
            "position" : position
        })
    
        
    @staticmethod
    def repulsion_sphere(particle, center, stiff, r0, rate=1):
        """
        A sphere that encloses the particle
        
        Parameters:
            particle (int): the particle that the force acts upon
            center ([float, float, float]): the center of the sphere
            stiff (float): stiffness of trap
            r0 (float): radius of sphere at t=0
            rate (float): the sphere's radius changes to r = r0 + rate*t
        """
        return({
            "type" : "sphere",
            "center" : center,
            "stiff" : stiff,
            "r0" : r0,
            "rate" : rate
        })
       
              
class SimFiles:
    def __init__(self, sim_dir):
        self.sim_dir = sim_dir
        if os.path.exists(self.sim_dir):
            self.file_list = os.listdir(self.sim_dir)
            self.parse_current_files()
    
    def parse_current_files(self):
        if os.path.exists(self.sim_dir):
            self.file_list = os.listdir(self.sim_dir)
        else:
            print('Simulation directory does not exsist')
            return None
        for file in self.file_list:
            if file == 'trajectory.dat':
                self.traj = os.path.abspath(os.path.join(self.sim_dir, file))
            elif file == 'last_conf.dat':
                self.last_conf = os.path.abspath(os.path.join(self.sim_dir, file))
            elif (file.endswith(('.dat'))) and not (file.endswith(('energy.dat'))) and not (file.endswith(('trajectory.dat'))) and not (file.endswith(('error_conf.dat'))):
                self.dat = os.path.abspath(os.path.join(self.sim_dir, file))
            elif (file.endswith(('.top'))):
                self.top = os.path.abspath(os.path.join(self.sim_dir, file))
            elif file == 'forces.json':
                self.force = os.path.abspath(os.path.join(self.sim_dir, file))
            elif file == 'input':
                self.input = os.path.abspath(os.path.join(self.sim_dir, file))
            elif file == 'input.json':
                self.input_js = os.path.abspath(os.path.join(self.sim_dir, file))
            elif file == 'observables.json':
                self.observables = os.path.abspath(os.path.join(self.sim_dir, file))
            elif file == 'run.sh':
                self.run_file = os.path.abspath(os.path.join(self.sim_dir, file))
            elif (file.startswith(('slurm'))):
                self.run_file = os.path.abspath(os.path.join(self.sim_dir, file))
            elif 'energy' in file:
                self.energy = os.path.abspath(os.path.join(self.sim_dir, file))
            elif 'com_distance' in file:
                self.com_distance = os.path.abspath(os.path.join(self.sim_dir, file))
