"""
Interface for forward flux sampling
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Union
from oxDNA_analysis_tools.distance import distance
from oxDNA_analysis_tools.bond_analysis import bond_analysis

from ..oxdna_simulation import Simulation


class Comparison(Enum):
    LT = "<"
    GT = ">"
    LEQ = "<="
    GEQ = ">="
    # no equals


# todo: definately not hardcode lol
ALLOWED_ORDER_PARAMETERS = [
    "mindistance",
    "bond"
]


@dataclass(frozen=True)
class OrderParameter:
    name: str = field()
    order_parameter: str = field()  # specific set of options in oxdna
    pairs: list[tuple[int, int]] = field()  # list of pairs of residue indices

    def __post_init__(self):
        if self.order_parameter not in ALLOWED_ORDER_PARAMETERS:
            raise Exception(f"Invalid order parameter {self.order_parameter}")

    def write(self, fp: Path):
        with fp.open("a+") as f:
            f.write("{\n")
            f.write(f"\torder_parameter = {self.order_parameter}\n")
            f.write(f"\tname = {self.name}\n")
            for (n, (base1, base2)) in enumerate(self.pairs):
                f.write(f"\tpair{n + 1} = {base1}, {base2}\n")
            f.write("}\n")


def write_order_params(op_file_name: Path, *args):
    for op in args:
        op.write(op_file_name)


@dataclass(frozen=True)
class FFSInterface:
    """
    Interface for forward flux sampling
    An interface is defined by some order parameter having a defined relation to a value
    A simulation passes through an interface simulation.orderparameter [compare] val
    changes from False to True

    """

    # name of parameter which is used to define this interface
    op: OrderParameter = field()
    val: Any = field()
    compare: Comparison = field()

    def __invert__(self) -> FFSInterface:
        """
        Returns a copy of this interface, but with an inverted comparison operator
        """
        if self.compare == Comparison.LT:
            newop = Comparison.GEQ
        elif self.compare == Comparison.GT:
            newop = Comparison.LEQ
        elif self.compare == Comparison.LEQ:
            newop = Comparison.GT
        elif self.compare == Comparison.GEQ:
            newop = Comparison.LT
        else:
            raise Exception(f"unrecognized operator {self.compare}")

        return FFSInterface(self.op, self.val, newop)

    def flip(self) -> FFSInterface:
        """
        similar to __invert__ but instead of the logical opposite it reverses
        the direction of the boundry in phase-space. if that makes any sense
        """

        if self.compare == Comparison.LT:
            newop = Comparison.GT
        elif self.compare == Comparison.GT:
            newop = Comparison.LT
        elif self.compare == Comparison.LEQ:
            newop = Comparison.GEQ
        elif self.compare == Comparison.GEQ:
            newop = Comparison.LEQ
        else:
            raise Exception(f"unrecognized operator {self.compare}")

        return FFSInterface(self.op, self.val, newop)

    def test(self, val: Union[float, Simulation]) -> bool:
        if isinstance(val, float):
            if self.compare == Comparison.LT:
                return val < self.val
            elif self.compare == Comparison.GT:
                return val > self.val
            elif self.compare == Comparison.LEQ:
                return val <= self.val
            elif self.compare == Comparison.GEQ:
                return val >= self.val
            else:
                raise Exception(f"unrecognized operator {self.compare}")
        else:
            return self.test(self.op.compute_value(val))


@dataclass(frozen=True)
class Condition:
    # condition name, for writing a file
    condition_name: str = field()

    # or-deliniated interfaces
    interfaces: list[FFSInterface] = field()
    condition_type: str = field(default="or")

    def __post_init__(self):
        assert self.condition_type in ["or", "and"], f"Invalid condition type {self.condition_type}"

    def write(self, write_dir: Path):
        with (write_dir / self.file_name()).open("w") as f:
            f.write(f"action = stop_{self.condition_type}\n")
            for n, interface in enumerate(self.interfaces):
                f.write(f"condition{n + 1} = " + "{\n" +
                        f"{interface.op.name} {interface.compare.value} {interface.val}" +
                        "\n}\n")

    def file_name(self) -> str:
        return f"{self.condition_name}.txt"

    def get_order_params(self) -> list[OrderParameter]:
        return order_params(*self.interfaces)


def order_params(*args: FFSInterface) -> list[OrderParameter]:
    """
    lists all order parameters used in the interfaces passed as params
    """

    ops = []
    op_names = set()  # use name set to avoid pass-by-value bullshit
    for interface in args:
        if interface.op.name not in op_names:
            op_names.add(interface.op.name)
            ops.append(interface.op)
    return ops
