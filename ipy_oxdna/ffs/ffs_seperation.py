from typing import Any, Union

import logging
import time, random as rnd
import shutil, glob
from multiprocessing import Process, Lock, Value
from pathlib import Path


from .base_flux_sampler import BaseFluxSampler, success_pattern, read_output
from ..oxdna_simulation import Simulation
from ..oxlog import OxLogHandler
from .ffs_interface import FFSInterface, OrderParameter, Condition, write_order_params, order_params

# !/usr/bin/env python

'''
Forward Flux sampling: Flux generator a-la-Tom

Flavio
'''


class SeperationFluxer(BaseFluxSampler):
    order_params: list[OrderParameter]

    # interfaces
    lambda_fail: FFSInterface
    lambda_0: FFSInterface
    lambda_neg1: FFSInterface
    lambda_s: FFSInterface

    # ---- conditions------
    # condition where system either crosses the lambda_-1 interface going backwards or crosses the lambda_{0} interface
    # previously called "both" but renamed bc that's not what "both" means
    pass_or_fail: Condition
    # apart-forward. condition where the system crosses the lambda_{-1} interface going forward
    apart_fw: Condition
    # condition where the system either fails (crosses lambda_-1 going backwards) or succeds (all bonds totally dissociated)
    fail_or_success: Condition

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
        # make flux directory
        (self.tld() / "ffs_flux").mkdir()

        self.set_tld(self.tld() / "ffs_flux")
        self.set_tld()
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

        # print >> sys.stderr, "nstarted: %d, nsuccesses: %d success_prob: %g" % (nstarted, nsuccesses, nsuccesses/float(nstarted))
        main_log.info("terminating processes")

        main_log.info(f"Main: nsuccesses: {self.success_count.value - self.initial_success_count} in this run")

        # final computation of the flux
        stime = 0
        confs = glob.glob(success_pattern + '*')
        for conf in confs:
            with open(conf, 'r') as f:
                t = int(f.readline().split('=')[1])
                stime += t
        if len(confs):
            main_log.info(
                f"average number of timesteps taken to reach a success (including possibly previous"
                f" runs with the same pattern) (aka 1./flux): {float(stime) / len(confs)}")
            main_log.info(f"initial flux (includes previous runs if they were there): {len(confs) / float(stime)}")
        else:
            main_log.info("No confs generated!!!")

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
        sim.build()

        sim.input.swap_default_input("ffs")
        sim.input["T"] = f"{self.T}C"
        sim.input["seed"] = seed
        assert sim.file_dir.exists()

        if ffs_coindition is not None:
            # write order parameters file
            write_order_params(sim.sim_dir / "op.txt", *ffs_coindition.get_order_params())
            # write ffs condition file
            ffs_coindition.write(sim.sim_dir)
            sim.input["ffs_file"] = ffs_coindition.file_name()
            sim.input["order_parameters_file"] = "op.txt"

        sim.input.modify_input(other_inputs)
        sim.make_sequence_dependant()

        return sim

    # this function does the work of running the simulation, identifying a
    # success or a failure, and taking appropriate actions
    def ffs_process(self, idx: int, plogger: logging.Logger):

        # the seed is the index + initial seed, and the last_conf has an index as well
        seed = self.initial_seed + idx
        myrng = rnd.Random()
        myrng.seed(seed)

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
            if eq_sim.oxpy_run.error_message:
                raise Exception(eq_sim.oxpy_run.error_message)

            plogger.info("equilibrated")

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


            # grab ffs values
            op_values = read_output(init_sim)
            complete_success = self.lambda_s.test(op_values[self.lambda_s.op.name])

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
                    self.tld() / f"p{idx}/sim{simcount - 1}",
                    self.tld() / f"p{idx}/sim{simcount}",
                    myrng.randint(1, 50000),
                    self.apart_fw
                )
                simcount += 1
                # run
                sim.oxpy_run.run(subprocess=False)
                plogger.info("Worker %d: reached lambda_{-1} going forwards" % idx)

                # ------- flux sample -------------
                # continue running simulation until we either fail or hit the lambda_{0} interface
                sim = self.make_ffs_simulation(
                    {
                        'refresh_vel': 0,
                        'restart_step_counter': 0,
                        "steps": 2e10
                    },
                    sim,
                    self.tld() / f"p{idx}/sim{simcount}",
                    myrng.randint(1, 50000),
                    self.pass_or_fail
                )
                simcount += 1
                # run

                sim.oxpy_run.run(subprocess=False)

                op_values = read_output(sim)
                success = self.lambda_0.test(op_values[self.lambda_0.op.name])
                failure = self.lambda_neg1.test(op_values[self.lambda_neg1.op.name])
                # if we've had a success
                if success and not failure:
                    with self.success_lock:
                        # increment successes
                        self.success_count.value += 1
                        # copy last conf to working directory
                        shutil.copy(
                            f"{sim.sim_dir}/{sim.input.input_dict['lastconf_file']}",
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

                    # did we fully dissociate? gotta start over them
                    if complete_success:
                        shutil.copy(f"{sim.sim_dir}/{sim.input.input_dict['lastconf_file']}",
                                    "full_success" + str(self.success_count.value))
                        plogger.info(f"Worker {idx} has reached a complete success: restarting from equilibration")
                        break  # this breakes the innermost while cycle, which will also start next iteration of main loop
                    else:  # ok we're back in our begin state, can continue from that
                        plogger.info("Worker %d: crossed interface lambda_{-1} going backwards after success" % idx)
                elif failure and not success:  # idk what this means
                    plogger.info("Worker %d: crossed interface lambda_{-1} going backwards" % idx)
                else:
                    raise Exception("what the hell")



