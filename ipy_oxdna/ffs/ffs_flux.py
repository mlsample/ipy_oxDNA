#!/usr/bin/env python
'''
Forward Flux sampling flux generator a-la-Tom (I don't know who Tom is
Code written by Flavio Romano
Adapted by Josh Evans for ipy_oxdna
TODO: move to ipy_oxDNA
'''
from __future__ import annotations
import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Union, Generator

import networkx as nx
from ..oxdna_simulation import Simulation, SimulationManager
from .ffs_interface import FFSInterface, Condition

import time, random as rnd
from multiprocessing import Value

from pypatchy.util import get_spec_json


class FFSFluxer(SimulationManager):
    """
    I'm writing this specifically for a strand seperation
    TODO: generalize
    """
    input_file_name = 'input'  # name of input file
    input_file_params: dict
    logfilename = 'ffs.log'  # name of log file
    starting_conf = 'flux_initial.dat'  # name of starting conf
    success_pattern = './success_'  # output pattern for success file
    desired_success_count: int  # number of successful crossings of l0 desired

    # interface in phase-space at which we start
    lambda_neg1: FFSInterface

    # interface at which the entire process we're trying to test has completed
    # this is an optional parameter
    lambda_f: Union[FFSInterface, None]
    # first interface in the direction of success_interface in phase space
    lambda_0: FFSInterface

    workdir: Path  # working directory
    success_count: Value
    rng: rnd.Random

    ffs_parallel_queues: list[FFSSim]

    def __init__(self, workdir: Path,
                 desired_success_count: int,
                 start: FFSInterface, l0: FFSInterface,
                 success: Union[FFSInterface, None] = None,
                 input_file_params: Union[dict, str] = "ffs"):
        super().__init__(n_processes=ncpus)
        self.rng = rnd.Random()
        self.rng.seed(time.time())
        self.success_count = Value("nsuccesses", 0)
        self.desired_success_count = desired_success_count
        self.lambda_neg1 = start
        self.lambda_f = success
        self.lambda_0 = l0
        self.workdir = workdir

        # if the user has specified to load input file params from a source
        if isinstance(input_file_params, str):
            self.input_file_params = get_spec_json("input_files", input_file_params)["input"]
        else:
            self.input_file_params = input_file_params

    def run(self, join=False, gpu_mem_block=True, custom_observables=None, run_when_failed=False,
            cpu_run=False):

        super(FFSFluxer, self).run()

    def write_condition_files(self, targ_dir: Path):
        # TODO: better name
        # write done-or-success file
        # simulation has reached full-completion interface or
        # has reached the initial first interface
        Condition("fail-or-success", [
            self.lambda_f,  # simulation is complete
            ~self.lambda_neg1  # if we've fallen behind start interface
        ]).write(targ_dir)
        #
        Condition("start-state", [self.lambda_neg1]).write(targ_dir)
        #
        Condition("complete", [
            self.lambda_0,
            self.lambda_f
        ]).write(targ_dir)


class FFSSim(nx.DiGraph):
    """
    set of oxDNA simulations to forward-flux-sample
    this object should NOT be parallelized but rather should queue up simulations to
    be run by the FFSFluxer simulation manager

    post-equiliriation, the ffs simulation should have three stop conditions
    1. the simulation has passed behind the start condition in phase-space -> stop it and start over?
    2. the simulation has passed the success interface -> stop it and start over
    3. the simulation has passed the interface lambda _{0} ->
    """

    # context variables
    parentdir: Path
    rng: rnd.Random

    # unique identifier
    uid: int

    def __init__(self, parentdir: Path, rng: rnd.Random, uid: int):
        super().__init__()
        self.parentdir = parentdir
        self.rng = rng

        self.uid = uid

    def workdir(self) -> Path:
        return self.parentdir / f"n{self.uid}"  # todo: more descriptive

    def sim(self, i: int) -> Simulation:
        return self.nodes[i]["sim"]

    def equilibriate(self):
        """

        """
        assert len(self) == 0, "Attempting to equilibriate an existing... whatever we're calling this"
        self.add_node(0, sim=Simulation(self.parentdir, self.workdir() / "equilibriate"))
        self.sim(0).build()

    def build_run(self, predecessor: int):
        # create new simulation using predecessor's simulation dir as file dir
        # and create new simulation dir
        sim = Simulation(self.sim(predecessor).sim_dir,
                         self.workdir() / f"s{len(self)}")
        # specify
        sim.input_file({
            **self.sim(predecessor).input.input_dict,
            "seed": self.rng.randint(1, int(5e4))
        })
        # write condition fil
        # iter next index
        idx = len(self)
        assert idx not in self.nodes, "non-concurrancy issue?"
        self.add_edge(predecessor, idx)
        self.add_node(idx, sim=sim)
        return sim

    def build_interface_run(self,
                            predecessor: int,
                            ffs_condition: Condition) -> Simulation:
        sim = self.build_run(predecessor)
        sim.input_file({
            "ffs_file": f"{ffs_condition.condition_name}.txt"
        })
        # add condition attr to node
        self.nodes[len(self)-1]["ffs_condition"] = ffs_condition
        # write condition file to simulation directory
        ffs_condition.write(sim.sim_dir)
        sim.build()
        return sim


# class FluxFindWorker(Process):
#     """
#     TODO: I can definately integrate this much better w/ ipy_oxdna
#     """
#     logger: logging.Logger
#     success_counter: Value # global memory to keep track of successes
#     desired_success_count: int
#     myrng: rnd.Random
#     input_file_params: dict
#     def __init__(self,
#                  idx: int,
#                  logger: logging.Logger,
#                  global_successes: Value,
#                  desired_success_count: int,
#                  start_conf_dir: str,
#                  input_file_params: dict):
#         super().__init__()
#         self.success_counter = global_successes
#         self.process_idx = idx
#         self.logger = logger
#         self.desired_success_count = desired_success_count
#         self.input_file_params = input_file_params
#
#         # the seed is the index + initial seed, and the last_conf has an index as well
#         self.myrng = rnd.Random()
#         self.start_conf_dir = start_conf_dir
#
#     def seed_rng(self, seed_val: int):
#         self.myrng.seed(seed_val)
#
#     def equilibriate(self):
#         """
#         equilibriates the simulation by running a quick MD relax
#         """
#         # create relax directory
#         with tempfile.TemporaryDirectory() as working_directory:
#             # create a simulation in temporary directory
#             sim = Simulation(self.start_conf_dir, working_directory, self.input_file_params)
#             # make sequence dependant
#             sim.sequence_dependant()
#             sim.input_file({
#                 "seed": self.myrng.randint(1, 50000),
#                 "steps": 1e5,
#                 "refresh_vel": 1,
#             })
#             sim.build()
#
#     def find_flux(self):
#         """
#         single execution of the find-flux algotithm
#
#         """
#         # create a temporary directory
#         with tempfile.TemporaryDirectory() as working_directory:
#             # create a simulation in temporary directory
#             sim = Simulation(self.start_conf_dir, working_directory, self.input_file_params)
#             # make sequence dependant
#             sim.sequence_dependant()
#             sim.input_file({"seed": self.myrng.randint(1,50000)})
#
#             # initial equilibration
#             # open a file to handle the output
#             output = tf.TemporaryFile('r+', suffix=str(self.process_idx))
#
#             self.logger.log("Worker %d: equilibration started " % idx)
#             r = sp.call(command, stdout=output, stderr=sp.STDOUT)
#             assert (r == 0)
#             self.log("Worker %d: equilibrated " % idx)
#
#             # edit the command; we set to 0 the timer ONLY for the first time
#             # command = my_base_command + ['ffs_file=apart-bw.txt', 'restart_step_counter=1', 'seed=%d' % myrng.randint(1,50000)]
#             command = my_base_command + ['ffs_file=apart-or-success.txt', 'restart_step_counter=1',
#                                          'seed=%d' % myrng.randint(1, 50000)]
#
#             output.seek(0)
#
#             # here we run the command
#             # print command
#             r = sp.call(command, stdout=output, stderr=sp.STDOUT)
#             if r != 0:
#                 print("Error running program", file=sys.stderr)
#                 print("command line:", file=sys.stderr)
#                 txt = ''
#                 for c in command:
#                     txt += c + ' '
#                 print(txt, file=sys.stderr)
#                 print('output:', file=sys.stderr)
#                 output.seek(0)
#                 for l in output.readlines():
#                     print(l, end=' ', file=sys.stderr)
#                 output.close()
#                 sys.exit(-2)
#             # now we process the output to find out wether the run was a complete success
#             # (interface lambda_s reached) or a complete failure (interface lamda_f reached)
#             output.seek(0)
#             for line in output.readlines():
#                 words = line.split()
#                 if len(words) > 1:
#                     if words[1] == 'FFS' and words[2] == 'final':
#                         # print line,
#                         data = [w for w in words[4:]]
#             op_names = data[::2]
#             op_value = data[1::2]
#             op_values = {}
#             for ii, name in enumerate(op_names):
#                 op_values[name[:-1]] = float(op_value[ii][:-1])
#             complete_failure = eval('op_values["%s"] %s %s' % (lambda_f_name, '>', str(lambda_f_value)))
#             complete_success = eval('op_values["%s"] %s %s' % (lambda_s_name, lambda_s_compar, str(lambda_s_value)))
#             if (complete_success):
#                 log("Worker %d has reached a complete success: returning with the tail in between my legs");
#                 continue
#
#             self.logger.info("Worker %d: reached Q_{-2}..." % self.process_idx)
#             # now the system is far apart;
#
#             while self.success_counter.value < self.desired_success_count:
#                 # cross lamnda_{-1} going forwards
#                 output.seek(0)
#                 command = my_base_command + ['ffs_file=apart-fw.txt', 'seed=%d' % myrng.randint(1, 50000)]
#                 r = sp.call(command, stdout=output, stderr=sp.STDOUT)
#                 assert (r == 0)
#                 log("Worker %d: reached lambda_{-1} going forwards" % idx)
#
#                 # we hope to get to success
#                 output.seek(0)
#                 command = my_base_command + ['ffs_file=both.txt', 'seed=%d' % myrng.randint(1, 50000)]
#                 r = sp.call(command, stdout=output, stderr=sp.STDOUT)
#                 assert (r == 0)
#
#                 # now we process the output to find out wether the run was a success
#                 # (interface lambda_m reached) or a failure (interface lamda_f reached)
#                 output.seek(0)
#                 for line in output.readlines():
#                     words = line.split()
#                     if len(words) > 1:
#                         if words[1] == 'FFS' and words[2] == 'final':
#                             # print line,
#                             data = [w for w in words[4:]]
#                 op_names = data[::2]
#                 op_value = data[1::2]
#                 op_values = {}
#                 for ii, name in enumerate(op_names):
#                     op_values[name[:-1]] = float(op_value[ii][:-1])
#
#                 # now op_values is a dictionary representing the status of the final
#                 # configuration.
#                 # print op_values, 'op_values["%s"] %s %s' % (lambda_m_name, lambda_m_compar, str(lambda_m_value)), 'op_values["%s"] %s %s' % (lambda_f_name, lambda_f_compar, str(lambda_f_value))
#                 success = eval('op_values["%s"] %s %s' % (lambda_0_name, lambda_0_compar, str(lambda_0_value)))
#                 failure = eval('op_values["%s"] %s %s' % (lambda_f_name, '>', str(lambda_f_value)))
#
#                 # print "EEE", op_values, success, failure #, 'op_values["%s"] %s %s' % (lambda_0_name, lambda_0_compar, str(lambda_0_value)), 'op_values["%s"] %s %s' % (lambda_f_name, '<', str(lambda_f_value))
#
#                 if success and not failure:
#                     with success_lock:
#                         success_count.value += 1
#                         shutil.copy(my_conf, success_pattern + str(success_count.value))
#                     log("Worker %d: crossed interface lambda_{0} going forwards: SUCCESS" % idx)
#                     output.seek(0)
#                     # command = my_base_command + ['ffs_file=apart-bw.txt', 'restart_step_counter=1', 'seed=%d' % myrng.randint(1,50000)]
#                     command = my_base_command + ['ffs_file=apart-or-success.txt', 'restart_step_counter=1',
#                                                  'seed=%d' % myrng.randint(1, 50000)]
#                     r = sp.call(command, stdout=output, stderr=sp.STDOUT)
#                     assert (r == 0)
#                     output.seek(0)
#                     for line in output.readlines():
#                         words = line.split()
#                         if len(words) > 1:
#                             if words[1] == 'FFS' and words[2] == 'final':
#                                 # print line,
#                                 data = [w for w in words[4:]]
#                     op_names = data[::2]
#                     op_value = data[1::2]
#                     op_values = {}
#                     for ii, name in enumerate(op_names):
#                         op_values[name[:-1]] = float(op_value[ii][:-1])
#                     complete_failure = eval('op_values["%s"] %s %s' % (lambda_f_name, '>', str(lambda_f_value)))
#                     complete_success = eval('op_values["%s"] %s %s' % (lambda_s_name, lambda_s_compar, str(lambda_s_value)))
#                     if complete_success:
#                         shutil.copy(my_conf, "full_success" + str(success_count.value))
#                         logging.log("Worker %d has reached a complete success: restarting from equilibration" % idx)
#                         break  # this breakes the innermost while cycle
#                     else:
#                         logging.log("Worker %d: crossed interface lambda_{-1} going backwards after success" % idx)
#                 elif failure and not success:
#                     logging.log("Worker %d: crossed interface lambda_{-1} going backwards" % (idx))
#                 else:
#                     output.seek(0)
#                     # for l in output.readlines():
#                     #	print l,
#                     print(
#                         op_values)  # , 'op_values["%s"] %s %s' % (lambda_0_name, lambda_0_compar, str(lambda_0_value)), 'op_values["%s"] %s %s' % (lambda_f_name, '<', str(lambda_f_value))
#                     logging.log("Worker %d: UNDETERMINED" % (idx))
#                 # sys.exit()
#
#     def run(self) -> None:
#         """
#         Computes initial flux by finding success interfaces for lambda0
#         Runs in parallel
#         """
#
#         # continue looping while the
#         while self.success_counter.value < self.desired_success_count:
#
#
#         os.remove(my_conf)
#
# import threading
# import time
#
# class TimerThread(threading.Thread):
#     interval: int
#     success_count: multiprocessing.Value
#     logger: logging.Logger
#     start_time: float
#     running: bool
#     def __init__(self,  success_count: multiprocessing.Value, logger: logging.Logger, interval: int = 1e4):
#         super(TimerThread, self).__init__()
#         self.interval = interval
#         self.success_count = success_count
#         self.logger = logger
#         self.start_time = time.time()
#         self.running = True
#
#     def run(self):
#         while self.running:
#             time.sleep(self.interval)
#             elapsed_time = time.time() - self.start_time
#             current_success_count = self.success_count.value  # Assuming this is a multiprocessing.Value
#             if current_success_count > 0:
#                 time_per_success = elapsed_time / current_success_count
#                 self.logger.info(f"Timer: successes: {current_success_count}, time per success: {time_per_success:.2f} seconds")
#             else:
#                 self.logger.info("Timer: No successes yet")
#
#     def stop(self):
#         self.running = False


# define for executing module as a script
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add the arguments
    parser.add_argument('-n', '--num_successes', type=int, help='Number of successes')
    parser.add_argument('-s', '--seed', type=int, default=int(time.time()),
                        help='Initial seed for random number generation')
    parser.add_argument('-c', '--ncpus', type=int, default=1, help='Number of CPUs')
    parser.add_argument('-k', '--success_count', type=int, default=0, help='Initial success count')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    if args.verbose:
        print("Verbose mode enabled")

    # Here, replace the body of your original try-except block with direct access to args:
    # For example:
    initial_seed = args.seed
    verbose = args.verbose
    ncpus = args.ncpus
    initial_success_count = args.success_count - 1
    fluxer = FFSFluxer()
    fluxer.compute_flux(initial_seed, verbose, initial_success_count)
