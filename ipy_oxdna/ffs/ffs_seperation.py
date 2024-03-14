import argparse
import logging
import os, sys, getopt
import subprocess as sp
import time, random as rnd, tempfile as tf
import shutil, glob
from logging import getLogger
from multiprocessing import Process, Lock, JoinableQueue, Value, Array
from typing import IO

from ipy_oxdna.oxdna_simulation import Simulation
from ..oxlog import OxLogHandler
from .ffs_interface import FFSInterface, OrderParameter, Comparison, Condition, write_order_params

#!/usr/bin/env python

'''
Forward Flux sampling: Flux generator a-la-Tom

Flavio
'''

####################################
# edit here
####################################
# number of successes, can be changed via command line
desired_success_count = 100

# executable set-up
# precommand = 'mosrun -J 12'
# executable = '/home/josh/oxDNA/build/bin/oxDNA'
# input = 'input'
logfilename = 'ffs.log'
starting_conf = 'output_no_mismatch.dat'
success_pattern = './success_'

bonds = [
    (97, 37),
    (96, 38),
    (95, 39),
    (94, 40),
    (93, 41),
    (92, 42),
    (91, 43)
]

# construct order param for n bonds
native = OrderParameter("native", "bond", bonds)
# construct order parameter for distance
mindist = OrderParameter("distance", "mindistance", bonds)

# interfaces
# interface lambda_{-1}
lambda_f = FFSInterface(
    native,
    7,
    Comparison.GT
)

# interface lambda_{0}
lambda_0 = FFSInterface(
    native,
    5,  # 2 fewer bonds,
    Comparison.LEQ
)

# interface lambda_{success}
lambda_s = FFSInterface(
    mindist,
    5.,  # 5 distance units
    Comparison.GT
)

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

initial_seed = int(time.time())
verbose = False
ncpus = 1
initial_success_count = 0

# try:
#     for o in opts:
#         k, v = o
#         if k == '-n':
#             desired_success_count = int(v)
#         elif k == '-s':
#             initial_seed = int(v)
#         elif k == '-c':
#             ncpus = int(v)
#         elif k == '-v':
#             Verbose = True
#         elif k == '-k':
#             initial_success_count = int(v) - 1
#         else:
#             print("Warning: option %s not recognized" % (k), file=sys.stderr)
# except:
#     print("Error parsing options", file=sys.stderr)
#     sys.exit(3)
#
# log_lock = Lock()
# log_file = open(logfilename, 'w', 1)
#
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

success_lock = Lock()
success_count: Value = Value('i', initial_success_count)

# write the condition file
# condition_file = open('close.txt', 'w')
# condition_file.write("condition1 = {\n%s %s %s\n}\n" % (lambda_0_name, lambda_0_compar, str(lambda_0_value)))
# condition_file.close()

# condition_file = open('apart-bw.txt', 'w')
# condition_file.write("condition1 = {\n%s > %s\n}\n" % (lambda_f_name, str(lambda_f_value)))
# condition_file.close()

apart_or_success = Condition("apart-or-success",
                             [lambda_f, lambda_s],
                             "or")

# with open('apart-or-success.txt', 'w') as condition_file:
#     condition_file.write("action = stop_or\n")
#     # passed back across lambda_{-1} interface - more bonds than original
#     condition_file.write("condition1 = {\n%s > %s\n}\n" % (lambda_f_name, str(lambda_f_value)))
#     # lambda_{f} interface, reaction completion reached (whatever that means)
#     condition_file.write("condition2 = {\n%s %s %s\n}\n" % (lambda_s_name, lambda_s_compar, str(lambda_s_value)))

apart_fw = Condition(
    "apart_fw",  # "fw"?
    [~lambda_f]
)

# with open('apart-fw.txt', 'w') as condition_file:
#     # lambda_{-1} interface - more bonds than original
#     condition_file.write("condition1 = {\n%s <= %s\n}\n" % (lambda_f_name, str(lambda_f_value)))

both = Condition(
    "both",
    [lambda_0, lambda_f],
    "or"
)

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

# this function does the work of running the simulation, identifying a
# success or a failure, and taking appropriate actions
def ffs_process(lidx: int, plogger: logging.Logger):
    idx = lidx


    # the seed is the index + initial seed, and the last_conf has an index as well
    seed = initial_seed + idx
    myrng = rnd.Random()
    myrng.seed(seed)
    # myrng.random()  # pull one random number and discard, for some reason?
    # above line meant to replace `jumpahead` which was removed in Python 3. frankly i'm unsure what Flavio was even trying to do here
    # myrng.jumpahead(1)

    op_file_name = "op.txt"

    global success_count
    simcount = 0  # process-specific sim counter
    # outer while loop
    while success_count.value < desired_success_count:
        # ----- step 1: initial relax ---------
        eq_sim = Simulation(".", f"p{lidx}/sim{simcount}")
        simcount += 1
        eq_sim.input_file({
            # "conf_file": starting_conf,
            # "lastconf_file": "equilibriated.dat",
            "seed": myrng.randint(1, int(5e6)),
            "sim_type": "MD",
            "steps": 1e5,
            "refresh_vel": 1,
            'log_file': "log.dat",
            "print_energy_every": 1e5,
            "print_conf_every": 1e6,
            "no_stdout_energy": 1,
            "restart_step_counter": 0,
        })
        plogger.info(f"Worker %d: equilibration started " % idx)
        eq_sim.build()
        # do not use OxpyRun multiprocessing, since we're already in an mps thread
        eq_sim.oxpy_run.run_complete()

        plogger.info("Worker %d: equilibrated " % idx)

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

        # ---------- run to the beginning of our state ----------
        init_sim = Simulation(eq_sim.sim_dir, f"p{lidx}/sim{simcount}")
        simcount += 1
        init_sim.input_file({
            "print_energy_every": 1e5,
            "print_conf_every": 1e6,
            "no_stdout_energy": 1,
            "refresh_vel": 0,
            'log_file': "log.dat",
            "ffs_file": apart_or_success.file_name(),
            "order_parameters_file": "op.txt",
            "restart_step_counter": 1,
            "seed": myrng.randint(1, 50000)
        })
        # build simulation
        init_sim.build()
        # write order parameters file
        write_order_params(init_sim.sim_dir / op_file_name, mindist, native)
        # write ffs condition file
        apart_or_success.write(init_sim.sim_dir)
        # run
        eq_sim.oxpy_run.run_complete()

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

        op_values = read_output(init_sim)
        complete_success = lambda_s.test(op_values[lambda_s.op.name])

        # complete_success = eval('op_values["%s"] %s %s' % (lambda_s_name, lambda_s_compar, str(lambda_s_value)))
        if complete_success:
            plogger.info("Worker %d has reached a complete success: returning with the tail in between my legs")
            continue

        plogger.info("Worker %d: reached Q_{-2}..." % idx)
        # now the system is far apart;

        # now run simulations until done or something
        while success_count.value < desired_success_count:
            # ----- cross lambda_{-1} going forward -----------------------
            sim = Simulation(f"p{lidx}/sim{simcount-1}" f"p{lidx}/sim{simcount}")
            simcount += 1
            sim.input_file({
                # 'conf_file': % (my_conf), 'lastconf_file=%s' % (my_conf)]
                'print_energy_every':1e5,
                'print_conf_every': 1e6,
                'no_stdout_energy': 1,
                'refresh_vel': 0,
                'log_file': "log.dat",
                'restart_step_counter': 0,
                'ffs_file': apart_fw.file_name(),
                "order_parameters_file": "op.txt",
                'seed': myrng.randint(1, 50000)
            })
            # build simulation files
            sim.build()
            # build order params file
            write_order_params(sim.sim_dir / op_file_name, mindist, native)
            # write ffs file
            apart_fw.write(sim.sim_dir)
            # run
            eq_sim.oxpy_run.run_complete()

            # cross lamnda_{-1} going forwards
            # output.seek(0)
            # command = my_base_command + ['ffs_file=apart-fw.txt', 'seed=%d' % myrng.randint(1, 50000)]
            # r = sp.call(command, stdout=output, stderr=sp.STDOUT)
            # assert (r == 0)
            plogger.info("Worker %d: reached lambda_{-1} going forwards" % idx)

            # ------- literally cannot figure out what we're trying to do here -------------
            sim = Simulation(f"p{lidx}/sim{simcount - 1}" f"p{lidx}/sim{simcount}")
            simcount += 1
            sim.input_file({
                # 'conf_file': % (my_conf), 'lastconf_file=%s' % (my_conf)]
                'print_energy_every': 1e5,
                'print_conf_every': 1e6,
                'no_stdout_energy': 1,
                'refresh_vel': 0,
                'log_file': "log.dat",
                'restart_step_counter': 0,
                'ffs_file': both.file_name(),
                "order_parameters_file": "op.txt",
                'seed': myrng.randint(1, 50000)
            })
            # build simulation files
            sim.build()
            # build order params file
            write_order_params(sim.sim_dir / op_file_name, mindist, native)
            # write ffs file
            both.write(sim.sim_dir)
            # run
            eq_sim.oxpy_run.run_complete()

            # we hope to get to success
            # output.seek(0)
            # command = my_base_command + ['ffs_file=both.txt', 'seed=%d' % myrng.randint(1, 50000)]
            # r = sp.call(command, stdout=output, stderr=sp.STDOUT)
            # assert (r == 0)

            op_values = read_output(init_sim)
            success = lambda_0.test(op_values[lambda_0.op.name])
            failure = lambda_f.test(op_values[lambda_f.op.name])

            # op_values = read_output(output)

            # now op_values is a dictionary representing the status of the final
            # configuration.
            # success = eval('op_values["%s"] %s %s' % (lambda_0_name, lambda_0_compar, str(lambda_0_value)))
            # failure = eval('op_values["%s"] %s %s' % (lambda_f_name, '>', str(lambda_f_value)))

            # print "EEE", op_values, success, failure #, 'op_values["%s"] %s %s' % (lambda_0_name, lambda_0_compar, str(lambda_0_value)), 'op_values["%s"] %s %s' % (lambda_f_name, '<', str(lambda_f_value))

            # if we've had a success
            if success and not failure:
                with success_lock:
                    # increment successes
                    success_count.value += 1
                    # copy last conf to working directory
                    shutil.copy(
                        f"{sim.sim_dir}/{sim.input.input['lastconf_file']}",
                        f"{success_pattern + str(success_count.value)}.dat"
                    )

                plogger.info("Worker %d: crossed interface lambda_{0} going forwards: SUCCESS" % idx)

                # ---------------- continue back across lambda_{0} ----------------------

                # now that the simulation is past the lambda_{0} interface, we need to continue running it
                sim = Simulation(f"p{lidx}/sim{simcount - 1}" f"p{lidx}/sim{simcount}")
                simcount += 1
                sim.input_file({
                    # 'conf_file': % (my_conf), 'lastconf_file=%s' % (my_conf)]
                    'print_energy_every': 1e5,
                    'print_conf_every': 1e6,
                    'no_stdout_energy': 1,
                    'refresh_vel': 0,
                    'log_file': "log.dat",
                    'restart_step_counter': 1,
                    'ffs_file': apart_or_success.file_name(),
                    "order_parameters_file": "op.txt",
                    'seed': myrng.randint(1, 50000)
                })
                # build simulation files
                sim.build()
                # build order params file
                write_order_params(sim.sim_dir / op_file_name, mindist, native)
                # write ffs file
                apart_or_success.write(sim.sim_dir)
                # run
                eq_sim.oxpy_run.run_complete()

                op_values = read_output(init_sim)
                # complete_failure = lambda_f.test(op_values[lambda_f.op.name])
                complete_success = lambda_s.test(op_values[lambda_s.op.name])

                # output.seek(0)
                # command = my_base_command + ['ffs_file=apart-bw.txt', 'restart_step_counter=1', 'seed=%d' % myrng.randint(1,50000)]
                # command = my_base_command + ['ffs_file=apart-or-success.txt', 'restart_step_counter=1',
                #                              'seed=%d' % myrng.randint(1, 50000)]
                # r = sp.call(command, stdout=output, stderr=sp.STDOUT)
                # assert (r == 0)
                # op_values = read_output(output)
                # complete_failure = eval('op_values["%s"] %s %s' % (lambda_f_name, '>', str(lambda_f_value)))
                # complete_success = eval('op_values["%s"] %s %s' % (lambda_s_name, lambda_s_compar, str(lambda_s_value)))

                if complete_success:
                    shutil.copy(f"{sim.sim_dir}/{sim.input.input['lastconf_file']}",
                                "full_success" + str(success_count.value))
                    plogger.info(f"Worker {idx} has reached a complete success: restarting from equilibration")
                    break  # this breakes the innermost while cycle
                else:
                    plogger.info("Worker %d: crossed interface lambda_{-1} going backwards after success" % idx)
            elif failure and not success: # idk what this means
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


def read_output(init_sim: Simulation) -> dict[str, float]:
    """
    terrible code, but i'm making it its own terrible code method
    """
    try:
        data = False
        for line in init_sim.oxpy_run.sim_output.split("\n"):
            words = line.split()
            if len(words) > 1:
                # jesus fucking christ
                if words[1] == 'FFS' and words[2] == 'final':
                    data = [w for w in words[4:]]
        assert data is not False, "oxDNA output does not include requisite FFS information"
        op_names = data[::2]
        op_value = data[1::2]
        op_values = {}
        for ii, name in enumerate(op_names):
            op_values[name[:-1]] = float(op_value[ii][:-1])
        return op_values
    except Exception as e:
        raise Exception("Missing oxDNA simulation output!")

# timer function: it spits out things
def timer():
    logger = loghandler.spinoff("timer")
    logger.info(f"Timer started at {(time.asctime(time.localtime()))}")
    itime = time.time()
    while True: # arbrgfgfgwse
        time.sleep(10)
        now = time.time()
        with success_lock:
            ns = success_count.value - initial_success_count
            if ns > 1:
                logger.info(f"Timer: at { time.asctime(time.localtime())}: successes: {ns}, time per success: {(now - itime) / float(ns)} ({now - itime} sec)")
            else:
                logger.info(f"Timer: at {time.asctime(time.localtime())}: no successes yet (at {success_count.value})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add the arguments
    parser.add_argument('-n',
                        '--num_successes',
                        type=int,
                        help='Number of successes')
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        default=int(time.time()),
                        help='Initial seed for random number generation')
    parser.add_argument('-c',
                        '--ncpus',
                        type=int,
                        default=1,
                        help='Number of CPUs')
    parser.add_argument('-k',
                        '--success_count',
                        type=int,
                        default=0,
                        help='Initial success count')
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='Enable verbose output')

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    if args.verbose:
        print("Verbose mode enabled")

    # Here, replace the body of your original try-except block with direct access to args:
    # For example:
    initial_seed: int = args.seed
    verbose: bool = args.verbose
    ncpus: int = args.ncpus
    initial_success_count: int = args.success_count - 1

    # Proceed with the rest of your code logic using the parsed arguments

    processes = []
    loghandler = OxLogHandler("ffs")
    main_log = loghandler.spinoff("main")
    main_log.info(f"Main: STARTING new shooting for {desired_success_count}")
    desired_success_count += initial_success_count
    for i in range(ncpus):
        p = Process(target=ffs_process, args=(i, loghandler.spinoff(f"Worker{i}")))

        processes.append(p)

    tp = Process(target=timer)
    tp.start()
    main_log.info("Main: Starting processes...")
    for p in processes:
        p.start()

    main_log.info("Main: waiting for processes to finish")
    for p in processes:
        p.join()

    main_log.info("Main: Terminating timer")
    tp.terminate()  # terminate timer

    nsuccesses = success_count.value - initial_success_count
    # print >> sys.stderr, "nstarted: %d, nsuccesses: %d success_prob: %g" % (nstarted, nsuccesses, nsuccesses/float(nstarted))
    print("terminating processes", file=sys.stderr)

    main_log.info("Main: nsuccesses: %d in this run" % (success_count.value - initial_success_count))

    # final coputation of the flux
    stime = 0
    confs = glob.glob(success_pattern + '*')
    for conf in confs:
        with open(conf, 'r') as f:
            t = int(f.readline().split('=')[1])
            stime += t
    main_log.info(f"Main: average number of timesteps taken to reach a success (including possibly previous runs with the same pattern) (aka 1./flux): { float(stime) / len(confs)}")
    main_log.info(f"Main: initial flux (includes previous runs if they were there): {len(confs) / float(stime)}")
