"""
File created 29 March 2024 by josh to replace Observable class
"""
from __future__ import annotations

import copy
from typing import Union, Generator


# TODO: observable class!!

def distance(particle_1=None, particle_2=None, PBC=None, print_every=None, name=None):
    """
    Calculate the distance between two (groups) of particles
    """
    return Observable(name,
                      print_every,
                      ObservableColumn(
                          "distance",
                          particle_1=particle_1,
                          particle_2=particle_2,
                          PBC=PBC
                      )).export()
    # return ({
    #     "output": {
    #         "print_every": print_every,
    #         "name": name,
    #         "cols": [
    #             {
    #                 "type": "distance",
    #                 "particle_1": particle_1,
    #                 "particle_2": particle_2,
    #                 "PBC": PBC
    #             }
    #         ]
    #     }
    # })


def hb_list(print_every: Union[None, int] = None,
            name=None,
            only_count=None) -> dict:
    """
    Compute the number of hydrogen bonds between the specified particles
    """
    return Observable(name,
                      print_every,
                      ObservableColumn(
                          "hb_list",
                          order_parameters_file="hb_list.txt",
                          only_count=only_count
                      )).export()
    # return ({
    #     "output": {
    #         "print_every": print_every,
    #         "name": name,
    #         "cols": [
    #             {
    #                 "type": "hb_list",
    #                 "order_parameters_file": "hb_list.txt",
    #                 "only_count": only_count
    #             }
    #         ]
    #     }
    # })


def particle_position(particle_id=None, orientation=None, absolute=None, print_every=None, name=None)->dict:
    """
    Return the x,y,z postions of specified particles
    """
    return Observable(name,
                      print_every,
                      ObservableColumn(
                          "particle_position",
                          particle_id=particle_id,
                          orientation=orientation,
                          absolute=absolute
                      )).export()
    # return ({
    #     "output": {
    #         "print_every": print_every,
    #         "name": name,
    #         "cols": [
    #             {
    #                 "type": "particle_position",
    #                 "particle_id": particle_id,
    #                 "orientation": orientation,
    #                 "absolute": absolute
    #             }
    #         ]
    #     }
    # })


def potential_energy(print_every=None, split=None, name=None) -> dict:
    """
    Return the potential energy
    """
    return Observable(name,
                      print_every,
                      ObservableColumn(
                          "potential_energy",
                          split=f"{split}"
                      )).export()
    # return {
    #     "output": {
    #         "print_every": f'{print_every}',
    #         "name": name,
    #         "cols": [
    #             {
    #                 "type": "potential_energy",
    #                 "split": f"{split}"
    #             }
    #         ]
    #     }
    # }


def force_energy(print_every: Union[None, int] = None,
                 name: Union[None, str] = None,
                 print_group=None) -> dict:
    """
    Return the energy exerted by external forces
    """
    if print_group is not None:
        col = ObservableColumn("force_energy", print_group=f"{print_group}")
    else:
        col = ObservableColumn("force_energy")
    return Observable(name, print_every, col).export()
    # if print_group is not None:
    #     return ({
    #         "output": {
    #             "print_every": f'{print_every}',
    #             "name": name,
    #             "cols": [
    #                 {
    #                     "type": "force_energy",
    #                     "print_group": f"{print_group}"
    #                 }
    #             ]
    #         }
    #     })
    # else:
    #     return ({
    #         "output": {
    #             "print_every": f'{print_every}',
    #             "name": name,
    #             "cols": [
    #                 {
    #                     "type": "force_energy",
    #                 }
    #             ]
    #         }
    #     })


def kinetic_energy(print_every=None, name=None) -> dict:
    """
    Return the kinetic energy
    """
    return Observable(name, print_every, ObservableColumn("kinetic_energy")).export()
    # return ({
    #     "output": {
    #         "print_every": f'{print_every}',
    #         "name": name,
    #         "cols": [
    #             {
    #                 "type": "kinetic_energy"
    #             }
    #         ]
    #     }
    # })


class Observable:
    """
    Deprecated class for observable methods
    class was written by matt to organize methods that create observables
    methods are retined for backwards compatibility but now redirect
    """


    # deprecated: methods to create observable objects
    # going fwd pls call methods directly
    distance = distance

    hb_list = hb_list

    particle_position = particle_position

    potential_energy = potential_energy

    force_energy = force_energy

    kinetic_energy = kinetic_energy

    # TODO: multitype observables?

    # class member vars
    # all observables will have these characteristics
    _file_name: str  # name of file to print data to
    _print_every: int  # interval at which to print
    _cols: list[ObservableColumn]

    def __init__(self, name: str, print_every: int, *args: ObservableColumn):
        self._file_name = name
        self._print_every = print_every
        self._cols = [*args]

    def get_file_name(self) -> str:
        return self._file_name

    def set_file_name(self, newname: str):
        assert isinstance(newname, str)
        self._file_name = newname

    def get_print_every(self) -> int:
        return self._print_every

    def set_print_every(self, newval: int):
        assert isinstance(newval, int)
        self._print_every = newval

    def get_cols(self) -> Generator[ObservableColumn, None, None]:
        for col in self._cols:
            yield copy.deepcopy(col)

    def __len__(self):
        return len(self._cols)

    def __add__(self, other: ObservableColumn):
        return Observable(self.file_name, self.print_every, *self.cols)

    def export(self) -> dict:
        return {
            "output": {
                "print_every": self.print_every,
                "name": self.file_name,
                "cols": [
                    col.export() for col in self.cols
                ]
            }
        }

    file_name = property(get_file_name, set_file_name)
    print_every = property(get_print_every, set_print_every)
    cols = property(get_cols)

    # TODO: make callable on simulation or something?

class ObservableColumn:
    _type_name: str  # name of ovservable type (e.g. "distance", "hb_list", "PatchyBonds")
    col_attrs: dict[str, str]

    def __init__(self, name: str, **kwargs: str):
        self._type_name = name
        self.col_attrs = kwargs

    def get_type_name(self) -> str:
        return self._type_name

    def export(self) -> dict:
        return {
            "type": self.type_name,
            **self.col_attrs
        }

    type_name = property(get_type_name) # type name should not be settable
