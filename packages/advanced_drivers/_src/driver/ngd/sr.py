from typing import Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp


from netket import jax as nkjax
from netket.utils import mpi
from netket.utils.types import Union, Array, PyTree

from netket_pro._src import distributed as distributed
from netket_pro._src.api_utils.kwargs import ensure_accepts_kwargs


@partial(
    jax.jit,
    static_argnames=(
        "solver_fn",
        "mode",
        "collect_quadratic_model",
        "collect_gradient_statistics",
    ),
)
def _compute_sr_update(
    O_L,
    dv,
    *,
    diag_shift: Union[float, Array],
    solver_fn: Callable[[Array, Array], Array],
    mode: str,
    collect_quadratic_model: bool = False,
    collect_gradient_statistics: bool = False,
    proj_reg: Optional[Union[float, Array]] = None,
    momentum: Optional[Union[float, Array]] = None,
    old_updates: Optional[Array] = None,
    params_structure,
    weights,
    is_jac: Optional[PyTree] = None
):
    # We concretize the solver function to ensure it accepts the additional argument `dv`.
    # Typically solvers only accept the matrix and the right-hand side.
    solver_fn = ensure_accepts_kwargs(solver_fn, "dv")

    # if (momentum is not None) or (old_updates is not None) or (proj_reg is not None):
    #     raise ValueError("Not implemented")

    # (np, #ns) x (#ns) -> (np) - where the sum over #ns is done automatically
    # in sharding, while under MPI we need to do it manually with an allreduce_sum.
    grad, token = mpi.mpi_allreduce_sum_jax(O_L.T @ dv, token=None)

    # This does the contraction (np, #ns) x (#ns, np) -> (np, np).
    # When using sharding the sum over #ns is done automatically.
    # When using MPI we need to do it manually with an allreduce_sum.
    matrix, token = mpi.mpi_reduce_sum_jax(O_L.T @ O_L, root=0, token=token)
    matrix_side = matrix.shape[-1]  # * it can be ns or 2*ns, depending on mode

    if mpi.rank == 0:
        shifted_matrix = jax.lax.add(
            matrix, diag_shift * jnp.eye(matrix_side, dtype=matrix.dtype)
        )
        updates = solver_fn(shifted_matrix, grad, dv=dv)

        # Some solvers return a tuple, some others do not.
        if isinstance(updates, tuple):
            updates, info = updates
        else:
            info = {}

        updates = updates.reshape(mpi.n_nodes, -1)
        updates, token = mpi.mpi_scatter_jax(updates, root=0, token=token)
    else:
        updates = jnp.zeros((int(matrix_side / mpi.n_nodes),), dtype=jnp.float64)
        updates, token = mpi.mpi_scatter_jax(updates, root=0, token=token)
        info = None

    if info is None:
        info = {}

    if collect_quadratic_model:
        info.update(_compute_quadratic_model_sr(matrix, grad, updates))

    # If complex mode and we have complex parameters, we need
    # To repack the real coefficients in order to get complex updates
    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        num_p = updates.shape[-1] // 2
        updates = updates[:num_p] + 1j * updates[num_p:]

    if is_jac is not None or collect_gradient_statistics: 
        info.update(
            _compute_snr_derivative(
            O_L, dv, grad, weights, is_jac, token=token
        )
        )
    updates, token = distributed.allgather(updates, token=token)
    return updates, old_updates, info


@partial(
    jax.jit,
    static_argnames=("mode",),
)
def _compute_gradient_statistics_sr(
    O_L: Array,
    dv: Array,
    grad: Array,
    mode: str,
    params_structure,
    token=None,
):
    grad_var, token = mpi.mpi_allreduce_sum_jax(O_L.T**2 @ dv**2, token=token)
    N_mc = O_L.shape[0] * mpi.n_nodes
    num_p = grad.shape[-1] // 2
    grad_var = grad_var * N_mc - grad**2
    jax.debug.print('{x}', x=grad_var[num_p:])
    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        num_p = grad.shape[-1] // 2
        grad = grad[:num_p] + 1j * grad[num_p:]
        grad_var = grad_var[:num_p] + 1j * grad_var[num_p:]

        grad, token = distributed.allgather(grad, token=token)
        grad_var, token = distributed.allgather(grad_var, token=token)
        # return {"gradient_mean": grad, "gradient_variance": grad_var}
    return {'snr': jnp.mean(jnp.sqrt(jnp.abs(grad)**2/grad_var))}

@jax.jit
def _compute_snr_derivative(
    O_L: Array,
    dv: Array,
    grad: Array,
    weights,
    is_jac: PyTree = None,
    token=None
):
    """
    Computes the gradient of the snr, and additional info if asked
    
    """
    grad_var, token = mpi.mpi_allreduce_sum_jax(O_L.T**2 @ dv**2, token=token)
    N_mc = O_L.shape[0] * mpi.n_nodes
    num_p = grad.shape[-1] // 2
    grad_var = grad_var * N_mc - grad**2

    weights2 = jnp.stack([jnp.real(weights), jnp.imag(weights)], axis=-1)
    weights = jax.lax.collapse(weights2, 0, 2)
    dv_rw = weights * dv
    # grad_var = (O_L.T**2) @ (dv**2) * N_mc - (O_L.T@dv)**2
    grad_var = (O_L.T**2) @ (dv**2) * N_mc - 2 * (O_L.T@dv_rw)*(O_L.T@dv) + (O_L.T@dv)**2
    snr = jnp.abs(grad)/jnp.sqrt(grad_var)
    snr, token = distributed.allgather(snr,token=token)
    
    if is_jac is not None:
        is_jac = jax.tree_util.tree_map(
            lambda x: jax.lax.collapse(jnp.stack([jnp.real(x), jnp.imag(x)], axis=-1), 0, 2),
            is_jac
        )
        grad_v = jax.tree_util.tree_map(
            lambda x: mpi.mpi_allreduce_sum_jax((O_L.T**2) @ (x * dv**2) * N_mc - 2*grad*(O_L.T @ (x*dv)))[0],
            is_jac)
        
    
        snr_for_grad = 1/2 * jnp.abs(grad)/(grad_var)**(3/2)
        grad_snr = jax.tree_util.tree_map(lambda g : jnp.mean(g * snr_for_grad, axis=-1), grad_v) #(N_Pis, 2N_p) then (N_Pis,) after mean 
        
        grad_snr = jax.tree_util.tree_map(lambda x : distributed.allgather(x,token=token)[0], grad_snr)
        return {"grad_snr": grad_snr, 'snr': jnp.mean(snr)}
    
    else:
        return {'snr': jnp.mean(snr)}

@jax.jit
def _compute_quadratic_model_sr(
    S: Array,  # (np, np)
    F: Array,  # (np, 1)
    δ: Array,  # (np, 1)
):
    r"""
    Computes the linear and quadratic terms of the SR update.
    The quadratic model reads:
    .. math::
        M(\delta) = h(\theta) + \delta^T \nabla h(\theta) + \frac{1}{2} \delta^T S \delta
    where :math:`h(\theta)` is the function to minimize. The linear and quadratic terms are:
    .. math::
        \text{linear_term} = \delta^T F
    .. math::
        \text{quadratic_term} = \delta^T S \delta

    Args:
        S: The quantum geometric tensor.
        F: The gradient of the function to minimize.
        δ: The proposed update.

    Returns:
        A dictionary with the linear and quadratic terms.
    """
    # (1, np) x (np, 1) -> (1, 1)
    linear = F.T @ δ

    # (1, np) x (np, np) x (np, 1) -> (1, 1)
    quadratic = δ.T @ (S @ δ)

    return {"linear_term": linear, "quadratic_term": quadratic}
