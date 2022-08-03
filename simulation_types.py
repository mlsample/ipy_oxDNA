from oxdna_json_utils import *
from oxdna_writing_utils import *
from simulation_interfaces import *
from forces import Force
import numpy as np
import os
import shutil


class Simulation:
    @staticmethod
    def equilibration(file_dir=None, sim_dir=None, input_json_dict=None, run_file=None, auto_run=False):
        #Make eq dir
        if not os.path.exists(sim_dir):
            os.mkdir(sim_dir)

        #Write dat and top files
        dat, top = get_last_conf_top(file_dir)
        write_dat_top(file_dir, sim_dir, dat, top)

        #Write input file
        write_input(sim_dir, input_json_dict)

        #Write run file
        if run_file is not None:
            sim_type = 'eq'
            write_run_file(sim_dir, sim_type, run_file)

        #Write sequence dependant parameters file
        if input_json_dict['use_average_seq'] == 'false':
            shutil.copy(os.path.join(file_dir, 'oxDNA2_sequence_dependent_parameters.txt'), sim_dir)

        if auto_run:
            os.chdir(sim_dir)
            p = run_oxpy_manager(sim_dir)
            return p
        return None

    @staticmethod
    def force_equilibration(file_dir=None, sim_dir=None, all_forces=None, input_json_dict=None, run_file=None, auto_run=False):
        #make eq dir
        if not os.path.exists(sim_dir):
            os.mkdir(sim_dir)

        #write dat and top files
        dat, top = get_last_conf_top(file_dir)
        write_dat_top(file_dir, sim_dir, dat, top)

        #write force file
        for f_number, force in enumerate(all_forces):
            if force['external_force'] == 'trap':
                mutual_trap_json(external_force=force['external_force'],
                                 particle=force['particle'],
                                 pos0=force['pos0'],
                                 k=force['k'],
                                 pbc=force['pbc'],
                                 rate=force['rate'],
                                 dir=force['dir'],
                                 f_number=f_number,
                                 write_path=os.path.join(sim_dir, f'forces.json')
                                 )
            elif force['external_force'] == 'com':
                com_force_json(external_force=force['external_force'],
                               p1=force['p1'],
                               p2=force['p2'],
                               k=force['k'],
                               r0=force['r0'],
                               pbc=force['pbc'],
                               rate=force['rate'],
                               f_number=f_number,
                               write_path=os.path.join(sim_dir, f'forces.json')
                               )
        #write input file
        write_input(sim_dir, input_json_dict)

        #write run file
        if run_file is not None:
            write_run_file(sim_dir, force['external_force'], run_file)

        #Write sequence dependant parameters file
        if input_json_dict['use_average_seq'] == 'false':
            shutil.copy(os.path.join(file_dir, 'oxDNA2_sequence_dependent_parameters.txt'), sim_dir)

        if auto_run:
            os.chdir(sim_dir)
            p = run_oxpy_manager(sim_dir)
            return p
        return None


    @staticmethod
    def com_pulling(file_dir=None, sim_dir=None, all_forces=None, n_windows=None, xmin=None, xmax=None, input_json_dict=None, steps_per_conf=None, run_file=None, auto_run=False):
        #Make eq dir
        if not os.path.exists(sim_dir):
            os.mkdir(sim_dir)

        #Write dat and top files
        dat, top = get_last_conf_top(file_dir)
        write_dat_top(file_dir, sim_dir, dat, top)

        steps = int(n_windows) * steps_per_conf
        force_rate = float(np.round((xmax - xmin) / steps, 12))
        steps = str(int(steps))
        input_json_dict["steps"] = steps
        input_json_dict['print_conf_interval'] = f'{int(steps_per_conf)}'
        input_json_dict['print_energy_every'] =  f'{int(steps_per_conf)}'
        #write force file
        for f_number, force in enumerate(all_forces):
            if force['external_force'] == 'trap':
                mutual_trap_json(external_force=force['external_force'],
                                 particle=force['particle'],
                                 pos0=force['pos0'],
                                 k=force['k'],
                                 pbc=force['pbc'],
                                 rate=force['rate'],
                                 dir=force['dir'],
                                 f_number=f_number,
                                 write_path=os.path.join(sim_dir, f'forces.json')
                                 )
            elif force['external_force'] == 'com':
                com_force_json(external_force=force['external_force'],
                               p1=force['p1'],
                               p2=force['p2'],
                               k=force['k'],
                               r0=force['r0'],
                               pbc=force['pbc'],
                               rate=str(force_rate),
                               f_number=f_number,
                               write_path=os.path.join(sim_dir, f'forces.json')
                               )
        #Write input file
        write_input(sim_dir, input_json_dict)

        #Write run file
        if run_file is not None:
            write_run_file(sim_dir, 'pulling', run_file)

        #Write sequence dependant parameters file
        if input_json_dict['use_average_seq'] == 'false':
            shutil.copy(os.path.join(file_dir, 'oxDNA2_sequence_dependent_parameters.txt'), sim_dir)

        if auto_run:
            os.chdir(sim_dir)
            p = run_oxpy_manager(sim_dir)
            return p

        return None
