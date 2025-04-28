from flax import linen as nn
from netket.utils.types import Array
import jax
import jax.numpy as jnp


class FlipExpSum(nn.Module):
    module: nn.Module

    @nn.compact
    def __call__(self, x: Array):
        # input has shape (-1, N_sites)
        # print(f"x.shape = {x.shape}")
        logpsi_plus = self.module(x)
        logpsi_minus = self.module(-x)
        # outputs have shape (-1,)
        logpsi_pm = jnp.stack(
            (logpsi_plus, logpsi_minus), axis=0
        )  # stack along a new zero axis, to be summed over
        # print(f"logpsi_plus.shape = {logpsi_plus.shape}, psi_minus.shape = {logpsi_minus.shape}, logpsi_pm.shape = {logpsi_pm.shape}")
        # print(f"logpsi_pm.shape = {logpsi_pm.shape}")
        # print(logpsi_pm)
        logpsi = jax.scipy.special.logsumexp(
            logpsi_pm, axis=0, b=1 / 2.0
        )  # compute log(0.5*(exp(logpsi_plus)+exp(logpsi_minus)))
        # print(f"logpsi.shape = {logpsi.shape}")
        return logpsi
