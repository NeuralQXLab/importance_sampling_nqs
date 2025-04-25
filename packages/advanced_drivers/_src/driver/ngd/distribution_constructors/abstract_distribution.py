from typing import Tuple
from netket.utils import struct
from netket.utils.types import PyTree, Callable


class AbstractDistribution(struct.Pytree, mutable=True):
    r"""
    Abstract distribution class used for sampling
    """

    q_variables: PyTree = struct.field(pytree_node=True, serialize=True)
    r"""
    Specific variales used to define the distribution
    """
    name: str = struct.field(pytree_node=False, serialize=True)
    r"""
    associated distribution name to keep track of the chain in the sampler
    """

    def __init__(self, q_variables: PyTree, name: str):
        self.q_variables = q_variables
        self.name = name

    def __call__(self, afun: Callable, variables: PyTree) -> Tuple[Callable, PyTree]:
        # returns a function to compute 1/2 \log f
        raise NotImplementedError
