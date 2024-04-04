import glob
import logging
import random
import shutil
from multiprocessing import Array, Process, Value, Lock
from pathlib import Path
from typing import Union, Any

from .base_flux_sampler import BaseFluxSampler, success_pattern, read_output
from .ffs_interface import FFSInterface, write_order_params, Condition
from ..oxdna_simulation import Simulation, find_top_file
from ..oxlog import OxLogHandler

undetermined_pattern = './undefin_'


class FFSShooter(BaseFluxSampler):
    keep_undetermined: bool

    # list of starting conf files
    starting_confs: list[str]

    # interface lambda_{-1} - fail interface
    lambda_f: FFSInterface

    # interface lambda_{n} - present interface?
    # lambda_n: FFSInterface

    # interface lambda_{n+1} - next interface?
    lambda_m: FFSInterface

    # success count is in superclass
    undetermined_count: Value
    undetermined_lock: Lock

    # array of successes from each start conf
    success_from: Array
    # array of attempts from each start conf
    attempt_from: Array

    def __init__(self,
                 num_successes: int,
                 desired_success_count: int,
                 T: float,
                 num_cpus: int = 1,
                 seed: int = 0):
        super().__init__(num_successes, desired_success_count, T, num_cpus, seed)
        self.keep_undetermined = False
        self.starting_confs = glob.glob(success_pattern + "*")
        assert len(self.starting_confs) > 0
        self.loghandler = OxLogHandler("ffs_shoot")
        main_log = self.loghandler.spinoff("main")
        main_log.info(f"Found {len(self.starting_confs)} starting confs")

        self.undetermined_count = Value("i", 0)
        self.undetermined_lock = Lock()

        self.success_from = Array('i', len(self.starting_confs))  # zeroed by default
        self.attempt_from = Array('i', len(self.starting_confs))  # zeroed by default

    def set_interfaces(self,
                       lambda_f: FFSInterface,
                    #    lambda_n: FFSInterface,
                       lambda_m: FFSInterface
                       ):
        self.lambda_f = lambda_f
        # self.lambda_n = lambda_n
        self.lambda_m = lambda_m

        # only one condition, can init it at runtime

    def run(self):
        (self.tld() / "shoot").mkdir()
        top_file_name = find_top_file(self.tld()).name
        shutil.copy(self.tld() / top_file_name, self.tld() / "shoot" / top_file_name)
        for conf in self.starting_confs:
            shutil.copy(self.tld() / conf, self.tld() / "shoot" / conf)
        self.set_tld(self.tld() / "shoot")
        processes = []
        main_logger = logging.getLogger("main")
        for i in range(self.ncpus):
            p = Process(target=self.ffs_process, args=(i, self.loghandler.spinoff(f"Worker{i}")))
            processes.append(p)

        tp = Process(target=self.timer)
        tp.start()
        main_logger.info("starting processes...")
        for p in processes:
            p.start()

        main_logger.info("waiting for processes to finish")
        for p in processes:
            p.join()

        main_logger.info("Terminating timer")
        tp.terminate()  # terminate timer

        nsuccesses = self.success_count.value - self.initial_success_count
        # print >> sys.stderr, "nstarted: %d, nsuccesses: %d success_prob: %g" % (nstarted, nsuccesses, nsuccesses/float(nstarted))
        main_logger.info("## log of successes probabilities from each starting conf")
        main_logger.info("conf_index nsuccesses nattempts prob")
        for k, v in enumerate(self.success_from):
            txt = f"{k}    {v}    {self.attempt_from[k]}   "
            if self.attempt_from[k] > 0:
                txt += f"{(float(v) / float(self.attempt_from[k]))}"
            else:
                txt += 'NA'
            main_logger.info(txt)
        main_logger.info("# SUMMARY")
        success_prob = nsuccesses / float(sum(self.attempt_from))
        main_logger.info(f"# nsuccesses: {nsuccesses} nattempts: {sum(self.attempt_from)} success_prob: {success_prob}"
                         f" undetermined: {self.undetermined_count.value}")

    def ffs_process(self, idx: int, plogger: logging.Logger):
        plogger.info(f"Worker {idx} started")
        sim_counter = 0
        while self.success_count.value < self.desired_success_count:
            # choose a starting configuration index
            conf_index: int = random.choice(list(range(0, len(self.starting_confs))))
            conf_file = Path(self.starting_confs[conf_index]).name

            plogger.info(f"Chose starting configuration {conf_file}")

            # iter attempt count
            self.attempt_from[conf_index] += 1
            sim_dir = self.tld() / f"p{idx}" / f"sim{sim_counter}"
            seed = self.initial_seed + sum(self.attempt_from)

            sim = self.make_ffs_simulation(conf_file,
                                           sim_dir,
                                           seed)

            sim.oxpy_run.run(subprocess=False)
            if sim.oxpy_run.error_message:
                raise Exception(sim.oxpy_run.error_message)
            sim_counter += 1

            op_values = read_output(sim)
            success = self.lambda_m.test(op_values[self.lambda_m.op.name])
            failure = self.lambda_f.test(op_values[self.lambda_f.op.name])

            if success and not failure:
                with self.success_lock:
                    self.success_count.value += 1
                    self.success_from[conf_index] += 1
                    shutil.copy(
                        f"{sim.sim_dir}/{sim.input.input_dict['lastconf_file']}",
                        f"{sim.file_dir}/shoot_success_{str(self.success_count.value)}.dat"
                    )
                    plogger.info(f"SUCCESS: worker {idx}: starting from conf_index {conf_index} and seed {seed}")
            elif not success and failure:
                # do else
                plogger.info(f"FAILURE: worker {idx}: starting from conf_index {idx} and seed {seed}")
            else:
                # do undetermined
                sim_log_file = sim.sim_dir / sim.input.input_dict["log_file"]
                with sim_log_file.open("r") as f:
                    txt = f.read()
                plogger.info(f"UNDETERMINED: worker {idx}: starting from conf_index {conf_index} and seed {seed}"
                             f"\n{txt}")
                with self.undetermined_lock:
                    self.undetermined_count.value += 1
                    if self.keep_undetermined:
                        shutil.copy(
                            f"{sim.sim_dir}/{sim.input.input_dict['lastconf_file']}",
                            f"{undetermined_pattern + str(self.undetermined_count.value)}.dat"
                        )
        plogger.info(f"Enough processes are started. Worker {idx} returning")

    def make_ffs_simulation(self,
                            start_conf: str,
                            sim_dir: Path,
                            seed: int,
                            ) -> Simulation:
        # todo: employ matt's defaults system when he writes it
        sim = Simulation(self.tld(), sim_dir)
        sim.build_sim.conf_file_name = start_conf
        sim.build()

        sim.input.swap_default_input("ffs")
        sim.input["T"] = f"{self.T}C"
        sim.input["seed"] = seed
        sim.input["restart_step_counter"] = 0
        sim.input["steps"] = 2e10 # as good as forever
        assert (self.tld() / start_conf).exists()

        # write order parameters file
        ffs_condition = Condition("shoot_condition", [self.lambda_f, self.lambda_m])
        write_order_params(sim.sim_dir / "op.txt", *ffs_condition.get_order_params())
        # write ffs condition file
        ffs_condition.write(sim.sim_dir)
        sim.input["ffs_file"] = ffs_condition.file_name()
        sim.input["order_parameters_file"] = "op.txt"

        sim.make_sequence_dependant()

        return sim
