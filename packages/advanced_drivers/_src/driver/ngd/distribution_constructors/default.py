from netket.utils.types import PyTree, Callable

from advanced_drivers._src.driver.ngd.distribution_constructors.abstract_distribution import (
    AbstractDistribution,
)


class default_distribution(AbstractDistribution):
    # default log-distribution to sample from
    # In the case of VMC, will return log \psi
    def __init__(
        self,
    ):
        self.q_variables = {}
        self.name = "default"

    def __call__(self, afun: Callable, variables: PyTree):
        return afun, variables
