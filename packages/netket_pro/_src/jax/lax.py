from functools import partial
import jax


@partial(jax.jit, static_argnums=1)
def reduce_xor(x, axes):
    if isinstance(axes, int):
        axes = (axes,)
    axes = tuple(i if i >= 0 else x.ndim + i for i in axes)
    return jax.lax.reduce_xor_p.bind(x, axes=axes)
