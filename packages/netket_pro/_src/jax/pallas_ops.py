from typing import Literal
from functools import partial

import jax.numpy as jnp
import numpy as np

import netket.jax as nkjax

import jax
from jax.experimental import pallas as pl


def vec_to_tril_pl_kernel(pars_ref, o_ref, *, nv: Literal[int], k: Literal[int]):
    """
    Pallas kernel to generate a upper triangular matrix from a linear array of parameters.
    """
    i, j = np.tril_indices(nv, k=k)
    o_ref[:, :] = jnp.zeros(o_ref.shape, dtype=o_ref.dtype)
    for k, (ii, jj) in enumerate(zip(i, j)):
        o_ref[ii, jj] = pars_ref[k]


def tril_to_vec_pl_kernel(matrix, out_vec, *, nv: Literal[int], k: Literal[int]):
    """
    Pallas kernel to generate a linear array of parameters from an upper triangular matrix.
    This is the reverse of the operation above.
    """
    i, j = np.tril_indices(nv, k=k)
    out_vec[:] = jnp.zeros(out_vec.shape, dtype=out_vec.dtype)
    for idx, (ii, jj) in enumerate(zip(i, j)):
        out_vec[idx] = matrix[ii, jj]


## The function here is vec_to_tril with a custom vjp because pallas kernels do not support vjp


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def vec_to_tril_ad(
    pars: jax.Array,
    nv: int,
    k: int = -1,
    interpret: bool = False,
) -> jax.Array:
    to_hermitian = pl.pallas_call(
        partial(vec_to_tril_pl_kernel, nv=nv, k=k),
        out_shape=jax.ShapeDtypeStruct((nv, nv), nkjax.dtype_real(pars.dtype)),
        interpret=interpret,
    )
    return to_hermitian(pars)


def vec_to_tril_fwd(
    pars: jax.Array,
    nv: int,
    k: int = -1,
    interpret: bool = False,
) -> jax.Array:
    to_hermitian = pl.pallas_call(
        partial(vec_to_tril_pl_kernel, nv=nv, k=k),
        out_shape=jax.ShapeDtypeStruct((nv, nv), nkjax.dtype_real(pars.dtype)),
        interpret=interpret,
    )
    # pass pars just to have access to the dtype
    return to_hermitian(pars), (pars)


def vec_to_tril_bwd(nv, k, interpret, pars, g):
    from_hermitian = pl.pallas_call(
        partial(tril_to_vec_pl_kernel, nv=nv, k=k),
        out_shape=jax.ShapeDtypeStruct(pars.shape, nkjax.dtype_real(pars.dtype)),
        interpret=interpret,
    )

    return (from_hermitian(g),)


vec_to_tril_ad.defvjp(vec_to_tril_fwd, vec_to_tril_bwd)


# The exposed function
@partial(jax.jit, static_argnames=("nv", "k", "interpret"))
def vec_to_tril(
    pars: jax.Array,
    nv: int,
    k: int = -1,
    interpret: bool = False,
) -> jax.Array:
    """
    Convert a linear array of parameters into a lower triangular matrix (including
    the diagonal) of size nv x nv.

    This can be used to generate a lower triangular or simmetric matrix while keeping
    only the parameters of the lower triangular part.

    If you write this naively, it will run ok on CPU but very slow on GPU, so this is
    a way to write an efficient Jastrow version on GPU.

    Args:
        pars: A linear array of parameters.
        nv: The size of the resulting square matrix.
        k: The diagonal offset. Default is -1, which means the lower tridiangola part without
            the diagonal. If k=0, the diagonal is included.
        interpret: If True, the pallas kernel is executed in interpret mode. This is
            slow, and only needed for debugging or running tests on CPU, and should not
            be used in production code.
    """

    if not len(np.tril_indices(nv, k=k)[0]) == pars.size:
        raise ValueError(
            "The number of parameters does not match the size of the matrix."
            "Expected: ",
            len(np.tril_indices(nv, k=k)[0]),
            "Got: ",
            pars.size,
        )

    to_hermitian = partial(vec_to_tril_ad, nv=nv, k=k, interpret=interpret)

    if jnp.issubdtype(pars.dtype, jnp.complexfloating):
        return jax.lax.complex(to_hermitian(pars.real), to_hermitian(pars.imag))
    else:
        return to_hermitian(pars)
