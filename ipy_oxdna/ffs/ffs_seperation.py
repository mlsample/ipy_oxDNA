from typing import Any, Union

import argparse
import logging
import time, random as rnd
import shutil, glob
from multiprocessing import Process, Lock, JoinableQueue, Value, Array
from pathlib import Path

import oxpy

from ..oxdna_simulation import Simulation, Observable
from ..oxlog import OxLogHandler
from .ffs_interface import FFSInterface, OrderParameter, Comparison, Condition, write_order_params, order_params

#!/usr/bin/env python
import os

'''
Forward Flux sampling: Flux generator a-la-Tom

Flavio
'''
success_pattern = './success_'


class SeperationFluxer:
    """
    TODO: make more generic or make abc
    """
    desired_success_count: int
    order_params: list[OrderParameter]

    success_lock: Lock
    success_count: Value

    # interfaces
    lambda_fail: FFSInterface
    lambda_0: FFSInterface
    lambda_neg1: FFSInterface
    lambda_s: FFSInterface

    initial_seed: int
    ncpus: int
    initial_success_count: int

    # ---- conditions------
    # condition where system either crosses the lambda_-1 interface going backwards or crosses the lambda_{0} interface
    # previously called "both" but renamed bc that's not what "both" means
    pass_or_fail: Condition
    # apart-forward. condition where the system crosses the lambda_{-1} interface going forward
    apart_fw: Condition
    # condition where the system either fails (crosses lambda_-1 going backwards) or succeds (all bonds totally dissociated)
    fail_or_success: Condition

    loghandler: logging
    T: int
    working_directory: Path

    def __init__(self,
                 num_successes: int,
                 desired_success_count: int,
                 T,
                 num_cpus: int = 1,
                 seed: int = 0):
        self.desired_success_count = desired_success_count

        self.initial_seed = int(time.time() + seed)
        # verbose = False
        self.ncpus = num_cpus
        self.initial_success_count = num_successes

        self.success_lock = Lock()
        self.success_count = Value('i', self.initial_success_count)
        self.T = T
        self.working_directory = Path.cwd()

    def tld(self) -> Path:
        return self.working_directory
    
    def set_tld(self, new_path: Path):
        self.working_directory = new_path

    def set_interfaces(self,
                       lambda_0: FFSInterface,
                       lambda_neg1: FFSInterface,
                       lambda_s: FFSInterface,
                       lambda_fail: Union[None, FFSInterface] = None):
        if lambda_fail is None:
            lambda_fail = ~lambda_neg1

        self.lambda_fail = lambda_fail
        self.lambda_0 = lambda_0
        self.lambda_neg1 = lambda_neg1
        self.lambda_s = lambda_s

        # condition where the system has either crossed the lambda_0 interface or gone back across the lambda_-1 interface
        self.pass_or_fail = Condition(
            "both",
            [lambda_0, lambda_fail],
            "or"
        )

        # strands seperate going forward
        self.apart_fw = Condition(
            "apart_fw",
            [lambda_neg1]
        )

        self.fail_or_success = Condition("apart-or-success",
                                     [lambda_neg1, lambda_s],
                                     "or")

        self.order_params = order_params(lambda_0, lambda_neg1, lambda_s, lambda_fail)

    def run(self):
        processes = []
        self.loghandler = OxLogHandler("ffs")
        main_log = self.loghandler.spinoff("main")
        main_log.info(f"Main: STARTING new shooting for {self.desired_success_count}")
        self.desired_success_count += self.initial_success_count
        for i in range(self.ncpus):
            p = Process(target=self.ffs_process, args=(i, self.loghandler.spinoff(f"Worker{i}")))

            processes.append(p)

        tp = Process(target=self.timer)
        tp.start()
        main_log.info("Main: Starting processes...")
        for p in processes:
            p.start()

        main_log.info("Main: waiting for processes to finish")
        for p in processes:
            p.join()

        main_log.info("Main: Terminating timer")
        tp.terminate()  # terminate timer

        nsuccesses = self.success_count.value - self.initial_success_count
        # print >> sys.stderr, "nstarted: %d, nsuccesses: %d success_prob: %g" % (nstarted, nsuccesses, nsuccesses/float(nstarted))
        main_log.error("terminating processes")

        main_log.info("Main: nsuccesses: %d in this run" % (self.success_count.value - self.initial_success_count))

        # final computation of the flux
        stime = 0
        confs = glob.glob(success_pattern + '*')
        for conf in confs:
            with open(conf, 'r') as f:
                t = int(f.readline().split('=')[1])
                stime += t
        if len(confs):
            main_log.info(
                f"average number of timesteps taken to reach a success (including possibly previous runs with the same pattern) (aka 1./flux): {float(stime) / len(confs)}")
            main_log.info(f"initial flux (includes previous runs if they were there): {len(confs) / float(stime)}")
        else:
            main_log.info("No confs generated!!!")

    # number of successes, can be changed via command line

    # executable set-up
    # precommand = 'mosrun -J 12'
    # executable = '/home/josh/oxDNA/build/bin/oxDNA'
    # input = 'input'
    # logfilename = 'ffs.log'
    # starting_conf = 'output_no_mismatch.dat'

    # interfaces


    ####################################
    # def usage():
    #     print('usage: %s %s' % (sys.argv[0], '[-n <num_sucesses>] [-s <seed>] [-c <ncpus>] [-k <success_count>]'),
    #           file=sys.stderr)
    #
    #
    # try:
    #     opts, files = getopt.gnu_getopt(sys.argv[1:], "n:s:c:k:v")
    # except getopt.GetoptError as err:
    #     usage()
    #     sys.exit(2)


    # try:
    #     for o in opts:
    #         k, v = o
    #         if k == '-n':
    #             desired_success_count = int(v)lambda_neg1
    # #
    # # def log(text):
    # #     log_lock.acquire()
    # #     log_file.write(text + '\n')
    # #     if Verbose:
    # #         print(text, file=sys.stdout)
    # #     log_lock.release()
    #
    #
    # # check that we can write to the success pattern
    # try:
    #     checkfile = open(success_pattern + '0', 'w')
    #     checkfile.close()
    #     os.remove(success_pattern + '0')
    # except:
    #     print("could not write to success_pattern", success_pattern, file=sys.stderr)
    #     sys.exit(3)


    # write the condition file
    # condition_file = open('close.txt', 'w')
    # condition_file.write("condition1 = {\n%s %s %s\n}\n" % (lambda_0_name, lambda_0_compar, str(lambda_0_value)))
    # condition_file.close()

    # condition_file = open('apart-bw.txt', 'w')
    # condition_file.write("condition1 = {\n%s > %s\n}\n" % (lambda_f_name, str(lambda_f_value)))
    # condition_file.close()

    # with open('apart-or-success.txt', 'w') as condition_file:
    #     condition_file.write("action = stop_or\n")
    #     # passed back across lambda_{-1} interface - more bonds than original
    #     condition_file.write("condition1 = {\n%s > %s\n}\n" % (lambda_f_name, str(lambda_f_value)))
    #     # lambda_{f} interface, reaction completion reached (whatever that means)
    #     condition_file.write("condition2 = {\n%s %s %s\n}\n" % (lambda_s_name, lambda_s_compar, str(lambda_s_value)))



    # with open('apart-fw.txt', 'w') as condition_file:
    #     # lambda_{-1} interface - more bonds than original
    #     condition_file.write("condition1 = {\n%s <= %s\n}\n" % (lambda_f_name, str(lambda_f_value)))



    # with open('both.txt', 'w') as condition_file:
    #     condition_file.write("action = stop_or\n")
    #     # reached lambda_{-1} interface - first interface passed,stop and record conf
    #     condition_file.write("condition1 = {\n%s %s %s\n}\n" % (lambda_0_name, lambda_0_compar, str(lambda_0_value)))
    #     # lambda_{-1} interface - more bonds than original
    #     condition_file.write("condition2 = {\n%s > %s\n}\n" % (lambda_f_name, str(lambda_f_value)))

    # # base command line; all features that need to be in the input file
    # # must be specified here
    # base_command = []
    # base_command += [executable, input, 'print_energy_every=1e5', 'print_conf_every=1e6', 'no_stdout_energy=1',
    #                  'refresh_vel=0', 'restart_step_counter=0']
    # base_command_string = ''.join(str(w) + ' ' for w in base_command)
    # log("Main: COMMAND: " + base_command_string)
    #
    # if not os.path.exists(starting_conf):
    #     log("the input file provided (%s) does not exist. Aborting" % (starting_conf))
    #     sys.exit(-3)
    #
    # if not os.path.exists(input):
    #     log("the input file provided (%s) does not exist. Aborting" % (input))
    #     sys.exit(-3)
    # # check of the input file. If it contains an entry for the log file, spit out an error
    # inf = open(input, 'r')
    # log_found = False
    # for line in inf.readlines():
    #     words = line.split("=")
    #     if len(words) >= 1:
    #         if words[0].lstrip().startswith("log_file"):
    #             log_found = True
    # if (log_found):
    #     print("\nERROR: This script does not work if \"log_file\" is set in the input file. Remove it! :)\n",
    #           file=sys.stderr)
    #     sys.exit(-2)
    # inf.close()
    #
    # if not os.path.exists(executable):
    #     log("the executable file provided (%s) does not exist. Aborting" % (executable))
    #     sys.exit(-3)

    # rnd.seed (initial_seed)

    # TODO: add param to write a message to the directory explaining what we're trying to do
    def make_ffs_simulation(self,
                    other_inputs: dict[str, Any], 
                    origin: Union[Simulation, Path], 
                    sim_dir: Path,
                    seed: int,
                    ffs_coindition: Union[Condition, None] = None
                    ) -> Simulation:
        # todo: employ matt's defaults system when he writes it
        sim = Simulation(origin if isinstance(origin, Path) else origin.sim_dir, sim_dir)
        assert sim.file_dir.exists()
        input_dict = {
            "interaction_type": "DNA2",
            "backend": "CPU",
            "sim_type": "FFS_MD",
            'log_file': "log.dat",
            "print_energy_every": 1e5,
            "print_conf_interval": 1e6,
            "no_stdout_energy": 1,
            "dt": 0.003,
            "verlet_skin": 0.05,
            "rcut": 2.0,
            "thermostat": "john",
            "newtonian_steps": 51,
            "diff_coeff": 1.25,
            "T": f"{self.T}C",
            "seed": seed,

            "salt_concentration": "1.0",
            "trajectory_file": "trajectory.dat",
            "energy_file": "energy.dat",
            "time_scale": "linear"
        }
        sim.build()

        if ffs_coindition is not None:
            # write order parameters file
            write_order_params(sim.sim_dir / "op.txt", *ffs_coindition.get_order_params())
            # write ffs condition file
            ffs_coindition.write(sim.sim_dir)
            input_dict["ffs_file"] = ffs_coindition.file_name()
            input_dict["order_parameters_file"] = "op.txt"

        input_dict.update(other_inputs)
        sim.input.clear()
        sim.input_file(input_dict)
        sim.sequence_dependant()

        return sim

    # this function does the work of running the simulation, identifying a
    # success or a failure, and taking appropriate actions
    def ffs_process(self, idx: int, plogger: logging.Logger):

        # the seed is the index + initial seed, and the last_conf has an index as well
        seed = self.initial_seed + idx
        myrng = rnd.Random()
        myrng.seed(seed)
        # myrng.random()  # pull one random number and discard, for some reason?
        # above line meant to replace `jumpahead` which was removed in Python 3. frankly i'm unsure what Flavio was even trying to do here
        # myrng.jumpahead(1)

        simcount = 0  # process-specific sim counter
        # outer while loop
        while self.success_count.value < self.desired_success_count:
            # ----- step 1: initial relax ---------
            # do this every time w/ a random seed to make sure we have different starting points for our simulation
            plogger.info("equilibration started")
            eq_sim = self.make_ffs_simulation({
                    "sim_type": "MD",
                    "steps": 1e5,
                    "refresh_vel": 1,
                    "print_energy_every": 1e2,
                    "restart_step_counter": 0
                },
                self.tld(),
                self.tld() / f"p{idx}/sim{simcount}",
                myrng.randint(1, int(5e6))
            )
            simcount += 1
            # do not use OxpyRun multiprocessing, since we're already in an mps thread
            eq_sim.oxpy_run.run(subprocess=False)

            plogger.info("equilibrated")

            # we copy the initial configuration here
            # my_conf = 'conf' + str(idx)
            # shutil.copy(starting_conf, my_conf)

            # edit the command to be launched
            # my_base_command = base_command + ['conf_file=%s' % (my_conf), 'lastconf_file=%s' % (my_conf)]

            # initial equilibration
            # command = my_base_command + ['seed=%d' % myrng.randint(1, 50000), 'sim_type=MD', 'steps=1e5', 'refresh_vel=1']
            # open a file to handle the output
            # output = tf.NamedTemporaryFile('r+', suffix=str(idx))

            # plogger.info(f"Worker %d: equilibration started " % idx)
            # r = sp.call(command, stdout=output, stderr=sp.STDOUT)
            # assert (r == 0)
            # log("Worker %d: equilibrated " % idx)

            # edit the command; we set to 0 the timer ONLY for the first time
            # command = my_base_command + ['ffs_file=apart-bw.txt', 'restart_step_counter=1', 'seed=%d' % myrng.randint(1,50000)]
            # command = my_base_command + ['ffs_file=apart-or-success.txt', 'restart_step_counter=1',
            #                              'seed=%d' % myrng.randint(1, 50000)]
            #
            # output.seek(0)

            # ---------- run for a bit for some reason? ---------
            # i'm like 70% sure this is to make sure we don't start out past lambda_{-1}
            # being a bit specific
            plogger.info("Running initial simulation?")
            init_sim = self.make_ffs_simulation({
                    "refresh_vel": 0,
                    "restart_step_counter": 1,
                    "steps": 1e10
                },
                eq_sim, 
                self.tld() / f"p{idx}/sim{simcount}",
                myrng.randint(1, 50000),
                self.fail_or_success
            )
            simcount += 1

            # # run

            # tried adding observable but this didn't work
            # init_sim.add_observable(Observable.hb_list(f"{1e5}", "bonds", True))
            init_sim.oxpy_run.run(subprocess=False)
            if init_sim.oxpy_run.error_message:
                raise Exception(init_sim.oxpy_run.error_message)

            # here we run the command
            # print command
            # r = sp.call(command, stdout=output, stderr=sp.STDOUT)
            # if r != 0:
            #     print("Error running program", file=sys.stderr)
            #     print("command line:", file=sys.stderr)
            #     txt = ''
            #     for c in command:
            #         txt += c + ' '
            #     print(txt, file=sys.stderr)
            #     print('output:', file=sys.stderr)
            #     output.seek(0)
            #     for l in output.readlines():
            #         print(l, end=' ', file=sys.stderr)
            #     output.close()
            #     sys.exit(-2)
            # op_values = read_output(output)
            # complete_failure = eval ('op_values["%s"] %s %s' % (lambda_f_name, '>', str(lambda_f_value)))

            # grab ffs values
            op_values = read_output(init_sim)
            complete_success = self.lambda_s.test(op_values[self.lambda_s.op.name])
            # complete_success = self.lambda_s.test(init_sim)

            # complete_success = eval('op_values["%s"] %s %s' % (lambda_s_name, lambda_s_compar, str(lambda_s_value)))
            # if the simumation fully dissociated, we need to start over b/c we can't get any confs to shoot with
            if complete_success:
                plogger.info("has reached a complete success, restarting")
                continue

            plogger.info("reached Q_{-2}...")
            # now the system is far apart;

            # now run simulations until done or something
            while self.success_count.value < self.desired_success_count:
                # ----- cross lambda_{-1} going forward -----------------------
                # construct new simulation from output of previous simulation
                sim = self.make_ffs_simulation(
                    {
                        'refresh_vel': 0,
                        'restart_step_counter': 0,
                        "steps": 2e10
                    },
                    # origin is either previous loop iteration or initial equilibriation
                    self.tld() / f"p{idx}/sim{simcount-1}",
                    self.tld() / f"p{idx}/sim{simcount}",
                    myrng.randint(1, 50000),
                    self.apart_fw
                )
                simcount += 1
                # run
                sim.oxpy_run.run(subprocess=False)

                # cross lamnda_{-1} going forwards
                # output.seek(0)
                # command = my_base_command + ['ffs_file=apart-fw.txt', 'seed=%d' % myrng.randint(1, 50000)]
                # r = sp.call(command, stdout=output, stderr=sp.STDOUT)
                # assert (r == 0)
                plogger.info("Worker %d: reached lambda_{-1} going forwards" % idx)

                # ------- flux sample -------------
                # continue running simulation until we either fail or hit the lambda_{0} interface
                sim = self.make_ffs_simulation(
                    {
                        'refresh_vel': 0,
                        'restart_step_counter': 0,
                        "steps": 2e10
                    }
                    ,
                    sim, 
                    self.tld() / f"p{idx}/sim{simcount}",
                    myrng.randint(1, 50000),
                    self.pass_or_fail
                )
                simcount += 1
                # run

                sim.oxpy_run.run(subprocess=False)

                # we hope to get to success
                # output.seek(0)
                # command = my_base_command + ['ffs_file=both.txt', 'seed=%d' % myrng.randint(1, 50000)]
                # r = sp.call(command, stdout=output, stderr=sp.STDOUT)
                # assert (r == 0)

                op_values = read_output(sim)
                success = self.lambda_0.test(op_values[self.lambda_0.op.name])
                failure = self.lambda_neg1.test(op_values[self.lambda_neg1.op.name])

                # op_values = read_output(output)

                # now op_values is a dictionary representing the status of the final
                # configuration.
                # success = eval('op_values["%s"] %s %s' % (lambda_0_name, lambda_0_compar, str(lambda_0_value)))
                # failure = eval('op_values["%s"] %s %s' % (lambda_f_name, '>', str(lambda_f_value)))

                # print "EEE", op_values, success, failure #, 'op_values["%s"] %s %s' % (lambda_0_name, lambda_0_compar, str(lambda_0_value)), 'op_values["%s"] %s %s' % (lambda_f_name, '<', str(lambda_f_value))

                # if we've had a success
                if success and not failure:
                    with self.success_lock:
                        # increment successes
                        self.success_count.value += 1
                        # copy last conf to working directory
                        shutil.copy(
                            f"{sim.sim_dir}/{sim.input.input['lastconf_file']}",
                            f"{success_pattern + str(self.success_count.value)}.dat"
                        )

                    plogger.info("Worker %d: crossed interface lambda_{0} going forwards: SUCCESS" % idx)

                    # ---------------- continue back across lambda_{0} ----------------------
                    # now that the simulation is past the lambda_{0} interface, we need to continue running it
                    # run until simulation is fully dissociate or have the @ least starting bond count
                    sim = self.make_ffs_simulation(
                        {
                            'refresh_vel': 0,
                            'restart_step_counter': 1,                            
                            "steps": 2e10
                        },
                        sim,
                        self.tld() / f"p{idx}/sim{simcount}",
                        myrng.randint(1, 50000),
                        self.fail_or_success
                    )
                    simcount += 1
                    # run
                    eq_sim.oxpy_run.run(subprocess=False)

                    op_values = read_output(init_sim)
                    # complete_failure = lambda_f.test(op_values[lambda_f.op.name])
                    complete_success = self.lambda_s.test(op_values[self.lambda_s.op.name])

                    # output.seek(0)
                    # command = my_base_command + ['ffs_file=apart-bw.txt', 'restart_step_counter=1', 'seed=%d' % myrng.randint(1,50000)]
                    # command = my_base_command + ['ffs_file=apart-or-success.txt', 'restart_step_counter=1',
                    #                              'seed=%d' % myrng.randint(1, 50000)]
                    # r = sp.call(command, stdout=output, stderr=sp.STDOUT)
                    # assert (r == 0)
                    # op_values = read_output(output)
                    # complete_failure = eval('op_values["%s"] %s %s' % (lambda_f_name, '>', str(lambda_f_value)))
                    # complete_success = eval('op_values["%s"] %s %s' % (lambda_s_name, lambda_s_compar, str(lambda_s_value)))

                    # did we fully dissociate? gotta start over them
                    if complete_success:
                        shutil.copy(f"{sim.sim_dir}/{sim.input.input['lastconf_file']}",
                                    "full_success" + str(self.success_count.value))
                        plogger.info(f"Worker {idx} has reached a complete success: restarting from equilibration")
                        break  # this breakes the innermost while cycle, which will also start next iteration of main loop
                    else:  # ok we're back in our begin state, can continue from that
                        plogger.info("Worker %d: crossed interface lambda_{-1} going backwards after success" % idx)
                elif failure and not success:  # idk what this means
                    plogger.info("Worker %d: crossed interface lambda_{-1} going backwards" % idx)
                else:
                    raise Exception("what the hell")
                    # output.seek(0)
                    # for l in output.readlines():
                    #	print l,
                    # print(
                    #     op_values)  # , 'op_values["%s"] %s %s' % (lambda_0_name, lambda_0_compar, str(lambda_0_value)), 'op_values["%s"] %s %s' % (lambda_f_name, '<', str(lambda_f_value))
                    # log("Worker %d: UNDETERMINED" % (idx))
                # sys.exit()

        # os.remove(my_conf)

    # timer function: it spits out things
    def timer(self):
        logger = self.loghandler.spinoff("timer")
        logger.info(f"Timer started at {(time.asctime(time.localtime()))}")
        itime = time.time()
        while True: # arbrgfgfgwse
            time.sleep(10)
            now = time.time()
            with self.success_lock:
                ns = self.success_count.value - self.initial_success_count
                if ns > 1:
                    logger.info(f"Timer: at { time.asctime(time.localtime())}: successes: {ns}, time per success: {(now - itime) / float(ns)} ({now - itime} sec)")
                else:
                    logger.info(f"Timer: at {time.asctime(time.localtime())}: no successes yet (at {self.success_count.value})")


def read_output(init_sim: Simulation) -> dict[str, float]:
    """
    terrible code, but i'm making it its own terrible code method
    """
    data = False
    sim_log_file = init_sim.sim_dir / init_sim.input.input["log_file"]
    if not sim_log_file.exists():
        raise Exception("No simulation run output!")
    with sim_log_file.open("r") as f:
        for line in f:
            words = line.split()
            if len(words) > 1:
                # jesus fucking christ
                if words[1] == 'FFS' and words[2] == 'final':
                    data = [w for w in words[4:]]
    if data is False:
        raise Exception("oxDNA output does not include requisite FFS information")
    op_names = data[::2]
    op_value = data[1::2]
    op_values = {}
    for ii, name in enumerate(op_names):
        op_values[name[:-1]] = float(op_value[ii][:-1])
    return op_values

