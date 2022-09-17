import json as js
from copy import deepcopy


# def observables_json(p1, p2, write_path=None):
#     observables = {
#         "output_1": {
#             "print_every": "1e4",
#             "name": "hbond_count.txt",
#             "cols": [
#                 {
#                     "type": "hb_list",
#                     "order_parameters_file": "op.txt",
#                     "only_count": "true"
#                 }
#             ]
#         },
#         "output_2": {
#             "print_every": "1e4",
#             "name": "com_distances.txt",
#             "cols": [
#                 {
#                     "type": "distance",
#                     "particle_1": p1,
#                     "particle_2": p2
#                 }
#             ]
#         }
#     }
#     if write_path is not None:
#         with open(write_path, 'w') as f:
#             f.write(js.dumps(observables, indent=4))
#     return observables

def observables_json(p1, p2, write_path=None):
    observables = {
        "output_1": {
            "print_every": "1e4",
            "name": "com_distances.txt",
            "cols": [
                {
                    "type": "distance",
                    "particle_1": p1,
                    "particle_2": p2
                }
            ]
        }
    }
    if write_path is not None:
        with open(write_path, 'w') as f:
            f.write(js.dumps(observables, indent=4))
    return observables


def input_json(write_path=None):
    input_dict = {
        "interaction_type": "DNA2",
        "salt_concentration": "1.0",
        "sim_type": "MD",
        "backend": "CUDA",
        "backend_precision": "mixed",
        "use_edge": "1",
        "edge_n_forces": "1",
        "CUDA_list": "verlet",
        "CUDA_sort_every": "0",
        "max_density_multiplier": "10",
        "steps": "1e7",
        "ensemble": "nvt",
        "thermostat": "john",
        "T": "25C",
        "dt": "0.002",
        "verlet_skin": "0.2",
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
        "print_conf_interval": "100000",
        "print_energy_every": "100000",
        "time_scale": "linear",
        "max_io": "5",
        "external_forces": "0",
        "external_forces_file": "forces.txt",
        "external_forces_as_JSON": "true",
    }
    if write_path is not None:
        with open(write_path, 'w') as f:
            js.dump(input_dict, f)
    return input_dict


def mutual_trap_json(external_force=None, particle=None, pos0=None, k=None,  pbc=None, rate=None, dir=None, f_number=None, write_path=None):
    force_js = {
        f"force_{(f_number * 2) + 1}": {
            "type": external_force,
            "particle": particle,
            "pos0": pos0,
            "stiff": k,
            "PBC": pbc,
            "rate": rate,
            "dir" : dir
        }
    }
    if write_path is not None:
        if f_number == 0:
            with open(write_path, 'w') as f:
                f.write(js.dumps(force_js, indent=4))
        else:
            with open(write_path) as f:
                read_force_js = js.loads(f.read())
                read_force_js.update(force_js.items())
                with open(write_path, 'w') as f:
                    f.write(js.dumps(read_force_js, indent=4))


def com_force_json(external_force=None, p1=None, p2=None, k=None, r0=None, pbc=None, rate=None, f_number=0, write_path=None):
    force_js = {
        f"force_{(f_number * 2) + 1}": {
            "type": external_force,
            "com_list": p1,
            "ref_list": p2,
            "stiff": k,
            "r0": r0,
            "PBC": pbc,
            "rate": str(rate)
        },
        f"force_{(f_number * 2) + 2}": {
            "type": external_force,
            "com_list": p2,
            "ref_list": p1,
            "stiff": k,
            "r0": r0,
            "PBC": pbc,
            "rate": str(rate)
        }
    }
    if write_path is not None:
        if f_number == 0:
            with open(write_path, 'w') as f:
                f.write(js.dumps(force_js, indent=4))
        else:
            with open(write_path) as f:
                read_force_js = js.loads(f.read())

                read_force_js.update(force_js.items())
                with open(write_path, 'w') as f:
                    f.write(js.dumps(read_force_js, indent=4))
    return force_js
