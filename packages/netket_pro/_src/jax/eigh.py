from functools import partial

import jax
from jax import lax
import jax.numpy as jnp


"""
This algorithm comes from Sandro's notes about how to compute the AD derivatives
of the eigendecomposition of a degenerate matrix.

This fixes jnp.linalg.eigh, which is broken.
"""


def eigh(A: jax.Array, symmetrize: bool = False) -> tuple[jax.Array, jax.Array]:
    """
    :func:`jax.numpy.linalg.eigh` with AD that works for degenerate matrices.

    The standard :func:`~jax.numpy.linalg.eigh` work fine in the forward pass, but if you
    try to take derivatives of it you get wrong results (usually NaN, sometimes numerical noise).

    This happens because the standard derivation of the AD rule for the eigendecomposition
    (see for example `those notes <https://math.mit.edu/~stevenj/18.336/adjoint.pdf>`_ ) are
    derived assuming non-degenerate matrices, but one should consider that case as well.

    This is well known, however the computational cost of taking 'stable derivatives' in the case
    of degenerate matrices is fairly high compared to the unstable formulas, so it's ignored in
    the standard implementations.

    .. note::

        This function only works on real-valued matrices. We could make it work in general,
        but this has not yet been done.

    Args:
        A: a symmetric real matrix. If it is not symmetric
        symmetrize: A boolean, false by default. If true, we force symmetrize the matrix.

    """
    if jnp.issubdtype(A.dtype, jnp.complexfloating):
        raise ValueError(
            "eigh_custom only supports real matrices at the moment, even though"
            "we could generalise it to complex if needed."
        )

    return eigh_custom(symmetrize, A)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def eigh_custom(symmetrize: bool, A: jax.Array) -> tuple[jax.Array, jax.Array]:
    if symmetrize:
        A_symm = 0.5 * (A + jnp.transpose(A))
    else:
        A_symm = A
    e, U = jnp.linalg.eigh(A_symm)
    return e, U


def division_cutoff(epsilon, value):
    return lax.cond(
        jnp.abs(epsilon) > 10e-10,
        lambda eps, val: val / eps,
        lambda eps, val: 0.0,
        epsilon,
        value,
    )


def rescale_matrix_M_by_eigenvalues(M, eigs):
    L = M.shape[0]

    # Create indices grid for vectorized operations
    i_indices = jnp.arange(L)
    j_indices = jnp.arange(L)

    # Define a vectorized function to apply division_cutoff elementwise
    def compute_element(i, j):
        return division_cutoff(eigs[i] - eigs[j], M[i, j])

    # Use vmap to vectorize the computation over all pairs (i, j)
    compute_matrix = jax.vmap(
        jax.vmap(compute_element, in_axes=(None, 0)),  # Vectorize over j
        in_axes=(0, None),  # Vectorize over i
    )

    return compute_matrix(i_indices, j_indices)


def eigh_custom_fwd(symmetrize, A):
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    e, U = eigh_custom(symmetrize, A)
    return (e, U), (e, U)


def eigh_custom_bwd(symmetrize, res, out_bar):
    # Gets residuals computed in f_fwd
    # e, U = eigh_custom(A)
    e, U = res
    # covectors in the output space.
    ē, Ū = out_bar

    M = jnp.transpose(Ū) @ U

    M = rescale_matrix_M_by_eigenvalues(M, e)

    M = M + jnp.diag(ē)

    M̄ = U @ jnp.transpose(U @ M)

    if symmetrize:
        M̄ = 0.5 * (M̄ + jnp.transpose(M̄))

    return (M̄,)


eigh_custom.defvjp(eigh_custom_fwd, eigh_custom_bwd)
