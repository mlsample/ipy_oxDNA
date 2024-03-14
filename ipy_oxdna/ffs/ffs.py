"""
top-level class and methods for forward flux sampling
"""

from .ffs_interface import FFSInterface
from .state import State


class ForwardFluxSampler:
    """
    top-level forward-flux-sampling class
    """
    a: State
    b: State

    def __init__(self, a: State, b: State):
        self.a = a
        self.b = b

    def partition(self) -> tuple[FFSInterface]:
        """
        constructs a series of dividing interfaces which seperate states A and B
        """

