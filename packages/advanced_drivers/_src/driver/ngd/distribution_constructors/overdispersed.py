from netket.utils.types import PyTree, Callable, Array
import netket.jax as nkjax
import jax.numpy as jnp

import flax.core as fcore
from advanced_drivers._src.driver.ngd.distribution_constructors.abstract_distribution import (
    AbstractDistribution,
)
import jax

class overdispersed_distribution(AbstractDistribution):
    r"""
    Overdispersed distribution:
    .. math::
        |\psi(x)|^\alpha, \alpha \in [0,2]

    Can help overcome the problem of a peaked wavefunction.
    """

    def __init__(self, alpha: float = 2.0, name: str = "overdispersed"):
        r"""
        Initializes the distribution
        Args:
            alpha: float giving the power of $|\psi(x)|$
        """
        super().__init__(q_variables={"alpha": jnp.array([alpha])}, name=name)

    def __call__(self, afun: Callable, variables: PyTree):
        new_variables = fcore.copy(variables, self.q_variables)
        return nkjax.HashablePartial(aux_fun, afun), new_variables

    def compute_log_grad_c(self, afun: Callable, new_variables: PyTree, samples):
        variables, alpha = fcore.pop(new_variables, "alpha")
        log_mod =  jnp.real(afun(variables, samples))
        return {'alpha': log_mod - jnp.mean(log_mod)}
    
    def update_params(self, grad_snr, lr: float = 2e-1, clip: float = 0.1):
        update = jax.tree_util.tree_map(lambda x : jnp.clip(lr * x, -clip, clip), grad_snr)
        self.q_variables = jax.tree_util.tree_map(lambda x,y : x + y, self.q_variables, update)
    

class overdispersed_mixture_distribution(AbstractDistribution):
    r"""
    Mixture of of n overdispersed distributions:
    .. math::
        1/n \sum |\psi|^\alpha_k, \alpha_k \in [0,2]
    Could help stabilize the automatic tuning.
    """

    def __init__(self, n: int = 2, alpha: Array = None):
        r"""
        Initializes the distribution

        Args :
            n: number of components in the mixture
            alpha: 1d array of powers to use, defaults to jnp.linspace(0,2,n)
        """
        if alpha is None:
            self.q_variables = {"alpha": jnp.linspace(0, 2, n)}
        else:
            self.q_variables = {"alpha": jnp.linspace(0, 2, n)}
        self.name = "overdispersed_mixture"

    def __call__(self, afun: Callable, variables: PyTree):
        new_variables = fcore.copy(variables, self.q_variables)
        return nkjax.HashablePartial(aux_fun_mixture, afun), new_variables


def aux_fun(afun, new_variables, x):
    variables, alpha = fcore.pop(new_variables, "alpha")
    return (alpha / 2) * afun(variables, x)


def aux_fun_mixture(afun, alpha, new_variables, x):
    variables, alpha = fcore.pop(new_variables, "alpha")
    log_psi = afun(variables, x)
    return (1 / 2) * jnp.log(
        jnp.mean((jnp.exp(jnp.real(log_psi))[:, None]) ** alpha, axis=1)
    )

