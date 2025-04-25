import flax.core as fcore

from jax.scipy.special import logsumexp

from netket.jax import HashablePartial
from netket.utils.types import PyTree, Callable
from netket.operator import AbstractOperator, ContinuousOperator

from advanced_drivers._src.driver.ngd.distribution_constructors.abstract_distribution import (
    AbstractDistribution,
)


class linearoperator_distribution(AbstractDistribution):
    r"""
    Distribution multiplied by linear operator:
    .. math::
        |O\psi(x)|^\alpha, O \in \text{Op}(\mathcal{H})
    """

    def __init__(self, O: AbstractOperator, name: str = "Opsi"):
        r"""
        Initializes the distribution
        Args:
            O: Linear operator to apply
            name: Name of the distribution (default: "Opsi")
        """
        name = "default" if O is None else name
        super().__init__(q_variables={"operator": O}, name=name)

    def __call__(self, afun: Callable, variables: PyTree):
        O = self.q_variables["operator"]
        if O is None:
            return afun, variables

        new_variables = fcore.copy(variables, self.q_variables)
        return HashablePartial(_logpsi_O_fun, afun), new_variables


def _logpsi_O_fun(afun, new_variables, x, *args):
    """
    This should be used as a wrapper to the original apply function, adding
    to the `variables` dictionary (in model_state) a new key `operator` with
    a jax-compatible operator.
    """
    variables, O = fcore.pop(new_variables, "operator")

    if isinstance(O, ContinuousOperator):
        res = O._expect_kernel(afun, variables, x)
    else:
        xp, mels = O.get_conn_padded(x)
        xp = xp.reshape(-1, x.shape[-1])
        logpsi_xp = afun(variables, xp, *args)
        logpsi_xp = logpsi_xp.reshape(mels.shape).astype(complex)

        res = logsumexp(logpsi_xp, axis=-1, b=mels)
    return res
