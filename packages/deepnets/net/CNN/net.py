import flax.linen as nn
from typing import Any, Callable
from collections.abc import Sequence
import numpy as np
from jax import vmap
import jax.numpy as jnp
from jax.random import PRNGKey, normal
from jax.nn.initializers import he_normal
from jax.nn import relu
from flax.linen.linear import PrecisionLike


def get_scale(fn, out_features, dtype, N):
    x = normal(PRNGKey(0), (1000, out_features), dtype=dtype)

    def output_std_eq(scale):
        out = jnp.sum(jnp.log(jnp.abs(fn(x * scale))), axis=1)
        target_std = 0.3 * np.sqrt(N)
        return (jnp.std(out) - target_std) ** 2

    scale = jnp.arange(0, 1, 0.01)
    out = vmap(output_std_eq)(scale)
    arg = jnp.nanargmin(out)
    return scale[arg]


def final_actfn(x):
    """
    The activation function applied after the final layer of CNN
    f(x) = cosh(x) if x > 0
           |2 - cosh(x)| if x <= 0
    """
    return jnp.where(x > 0, jnp.cosh(x), jnp.abs(2 - jnp.cosh(x)))


class ConvReLU(nn.Module):
    depth: int
    features: int
    kernel_size: Sequence
    graph: Any
    final_actfn: Callable
    dtype: Any = None
    precision: PrecisionLike = None

    def setup(self):
        # if self.depth % 2:
        #    raise ValueError("'depth' should be a multiple of 2.")
        self.lattice_shape = self.graph.extent
        dtype = self.dtype
        size = np.prod(self.lattice_shape[:-1])
        out_features = size * self.features
        self.scale = get_scale(
            self.final_actfn, out_features, dtype, self.graph.n_nodes
        )

        layers = []
        len_num = len(str(self.depth - 1))
        for i in range(self.depth):
            layers.append(
                nn.Conv(
                    features=self.features,
                    kernel_size=self.kernel_size,
                    padding="CIRCULAR",
                    dtype=dtype,
                    param_dtype=dtype,
                    precision=self.precision,
                    kernel_init=he_normal(dtype=dtype),
                    name="layers_" + "0" * (len_num - len(str(i))) + str(i),
                )
            )

        self.layers = layers
        self.layernorm = nn.LayerNorm(dtype=dtype, use_bias=False, use_scale=False)

    def __call__(self, x):
        x = x.reshape(x.shape[0], *self.lattice_shape)
        # print(f"x shape after reshape {x.shape}")
        x = x[..., jnp.newaxis]  # add features dimension
        residual = x.copy()

        for i, layer in enumerate(self.layers):
            if i:
                x = self.layernorm(x)
                x = relu(x)
            else:
                x /= jnp.sqrt(2)
            # print(f"x shape before layer {x.shape}")
            x = layer(x)  # input should have shape(batch, spatial_dims,features)
            # print(f"x shape after layer {x.shape}")
            if i % 2:
                x += residual
                residual = x.copy()

        x *= self.scale / jnp.sqrt(len(self.layers) // 2)
        complex_out = jnp.prod(self.final_actfn(x), axis=(-1, -2, -3)).astype(complex)
        # print(f"complex_out shape {complex_out.shape}")
        return jnp.log(complex_out)
