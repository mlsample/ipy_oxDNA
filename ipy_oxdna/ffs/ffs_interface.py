"""
Interface for forward flux sampling
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


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
        if self.order_parameter not in [ALLOWED_ORDER_PARAMETERS]:
            raise Exception(f"Invalid order parameter {self.order_parameter}")

    def write(self, fp: Path):
        with fp.open("r+") as f:
            f.write("{\n")
            f.write(f"\torder_parameter = {self.order_parameter}\n")
            f.write(f"\tname = {self.name}\n")
            for (n, (base1, base2)) in enumerate(self.pairs):
                f.write(f"\tpair{n+1} = {base1}, {base2}")
            f.write("}")


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

    def __invert__(self):
        """
        Returns a copy of this interface, but with an inverted comparison operator
        """
        if self.compare == Comparison.LT:
            newop = Comparison.GEQ
        elif self.compare == Comparison.GT:
            newop = Comparison.LEQ
        elif self.compare == Comparison.LEQ:
            newop = Comparison.GT
        elif self.compare == Comparison.GT:
            newop = Comparison.LT
        else:
            raise Exception(f"unrecognized operator {self.compare}")

        return FFSInterface(self.op, self.val, newop)

    def test(self, val: float) -> bool:
        if self.compare == Comparison.LT:
            return self.val < val
        elif self.compare == Comparison.GT:
            return self.val > val
        elif self.compare == Comparison.LEQ:
            return self.val <= val
        elif self.compare == Comparison.GT:
            return self.val >= val
        else:
            raise Exception(f"unrecognized operator {self.compare}")


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
            for n, interface in self.interfaces:
                f.write(f"condition{n+1} = " + "{\n" +
                        f"{interface.name} {interface.compare} {interface.value}" +
                        "\n}\n")

    def file_name(self) -> str:
        return f"{self.condition_name}.txt"
