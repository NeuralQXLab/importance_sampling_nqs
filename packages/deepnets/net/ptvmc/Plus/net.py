import jax.numpy as jnp

from netket.utils.types import DType, Array

import jax
import flax.linen as nn

class Plus(nn.Module):
    r"""
    Network with a constant output to parametrize the completely polarized state.
    """
    @nn.compact
    def __call__(self, x):
        w = self.param('w', jax.nn.initializers.normal(), (1,), jnp.float32)
        N = x.shape[-1]
        return -N/2 * jnp.ones(x.shape[:-1])