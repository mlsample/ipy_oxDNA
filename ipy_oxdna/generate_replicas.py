from __future__ import annotations

import os
import queue
import shutil
from typing import Any

from ipy_oxdna.oxdna_simulation import Simulation


class GenerateReplicas:
    """
    Methods to generate multisystem replicas
    """

    systems: Any
    n_replicas_per_system: int
    file_dir_list: list[str]
    sim_dir_list: list[str]
    sim_list: list[Simulation]
    queue_of_sims: queue.Queue

    # TODO: init

    def multisystem_replica(self,
                            systems,
                            n_replicas_per_system,
                            file_dir_list,
                            sim_dir_list):
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
                sim_list.append(Simulation(file_dir, sim_dir))
        q2 = queue.Queue()
        for sim in sim_list:
            q2.put(sim)

        self.sim_list = sim_list
        self.queue_of_sims = q2

    def concat_single_system_traj(self, system, concat_dir='concat_dir') -> tuple[list[str], str]:
        "Concatenate the trajectory of multiple replicas"
        system_index = self.systems.index(system)

        start_index = self.n_replicas_per_system * system_index
        end_index = start_index + self.n_replicas_per_system

        system_specific_sim_list = self.sim_list[start_index:end_index]
        sim_list_file_dir = [str(sim.file_dir) for sim in system_specific_sim_list]
        sim_list_sim_dir = [sim.sim_dir for sim in system_specific_sim_list]
        concat_dir = os.path.abspath(os.path.join(sim_list_file_dir[0], concat_dir))
        if not os.path.exists(concat_dir):
            os.mkdir(concat_dir)

        with open(f'{concat_dir}/trajectory.dat', 'wb') as outfile:
            for f in sim_list_sim_dir:
                with open(f'{f}/trajectory.dat', 'rb') as infile:
                    outfile.write(infile.read())
        shutil.copyfile(system_specific_sim_list[0].sim_files.top, concat_dir + '/concat.top')
        return sim_list_file_dir, concat_dir

    def concat_all_system_traj(self):
        "Concatenate the trajectory of multiple replicas for each system"
        self.concat_sim_dirs = []
        self.concat_file_dirs = []
        for system in self.systems:
            file_dir, concat_dir = self.concat_single_system_traj(system)
            self.concat_sim_dirs.append(concat_dir)
            self.concat_file_dirs.append(file_dir)
