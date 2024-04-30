"""
File created 29 March 2024 by josh to replace Observable class
"""
from typing import Union


# TODO: observable class!!

def distance(particle_1=None, particle_2=None, PBC=None, print_every=None, name=None):
    """
    Calculate the distance between two (groups) of particles
    """
    return ({
        "output": {
            "print_every": print_every,
            "name": name,
            "cols": [
                {
                    "type": "distance",
                    "particle_1": particle_1,
                    "particle_2": particle_2,
                    "PBC": PBC
                }
            ]
        }
    })


def hb_list(print_every: Union[None, int] = None,
            name=None,
            only_count=None):
    """
    Compute the number of hydrogen bonds between the specified particles
    """
    return ({
        "output": {
            "print_every": print_every,
            "name": name,
            "cols": [
                {
                    "type": "hb_list",
                    "order_parameters_file": "hb_list.txt",
                    "only_count": only_count
                }
            ]
        }
    })


def particle_position(particle_id=None, orientation=None, absolute=None, print_every=None, name=None):
    """
    Return the x,y,z postions of specified particles
    """
    return ({
        "output": {
            "print_every": print_every,
            "name": name,
            "cols": [
                {
                    "type": "particle_position",
                    "particle_id": particle_id,
                    "orientation": orientation,
                    "absolute": absolute
                }
            ]
        }
    })


def potential_energy(print_every=None, split=None, name=None):
    """
    Return the potential energy
    """
    return {
        "output": {
            "print_every": f'{print_every}',
            "name": name,
            "cols": [
                {
                    "type": "potential_energy",
                    "split": f"{split}"
                }
            ]
        }
    }


def force_energy(print_every: Union[None, int] = None,
                 name: Union[None, str] = None,
                 print_group=None):
    """
    Return the energy exerted by external forces
    """
    if print_group is not None:
        return ({
            "output": {
                "print_every": f'{print_every}',
                "name": name,
                "cols": [
                    {
                        "type": "force_energy",
                        "print_group": f"{print_group}"
                    }
                ]
            }
        })
    else:
        return ({
            "output": {
                "print_every": f'{print_every}',
                "name": name,
                "cols": [
                    {
                        "type": "force_energy",
                    }
                ]
            }
        })


def kinetic_energy(print_every=None, name=None):
    """
    Return the kinetic energy
    """
    return ({
        "output": {
            "print_every": f'{print_every}',
            "name": name,
            "cols": [
                {
                    "type": "kinetic_energy"
                }
            ]
        }
    })
