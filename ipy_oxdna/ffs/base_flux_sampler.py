import time
from abc import ABC, abstractmethod
from multiprocessing import Lock, Value
from pathlib import Path
import logging
from typing import Union

from ipy_oxdna.ffs.ffs_interface import FFSInterface
from ipy_oxdna.oxdna_simulation import Simulation

success_pattern = './success_'


class BaseFluxSampler(ABC):
    T: float
    desired_success_count: int
    initial_seed: int
    ncpus: int
    initial_success_count: int

    success_lock: Lock
    success_count: Value

    loghandler: logging
    working_directory: Path

    def __init__(self,
                 num_successes: int,
                 desired_success_count: int,
                 T: float,
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

    @abstractmethod
    def set_interfaces(self, *args: tuple[Union[FFSInterface, None], ...]):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def ffs_process(self, idx: int, plogger: logging.Logger):
        pass

    # timer function: it spits out things
    def timer(self):
        logger = self.loghandler.spinoff("timer")
        logger.info(f"Timer started at {(time.asctime(time.localtime()))}")
        itime = time.time()
        while True:  # arbrgfgfgwse
            time.sleep(10)
            now = time.time()
            with self.success_lock:
                ns = self.success_count.value - self.initial_success_count
                if ns > 1:
                    logger.info(
                        f"Timer: at {time.asctime(time.localtime())}: successes: {ns}, time per success: {(now - itime) / float(ns)} ({now - itime} sec)")
                else:
                    logger.info(
                        f"Timer: at {time.asctime(time.localtime())}: no successes yet (at {self.success_count.value})")


def read_output(init_sim: Simulation) -> dict[str, float]:
    """
    terrible code, but i'm making it its own terrible code method
    """
    data = False
    sim_log_file = init_sim.sim_dir / init_sim.input.input_dict["log_file"]
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
