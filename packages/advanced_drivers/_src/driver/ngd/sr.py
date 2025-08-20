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
    is_jac: Optional[PyTree] = None,
):
    if momentum is not None:
        dv -= momentum * (O_L @ old_updates)
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
    if momentum is not None:
        updates += momentum * old_updates
        old_updates = updates
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
            _compute_snr_derivative(O_L, dv, grad, weights, is_jac, token=token)
        )
    updates, token = distributed.allgather(updates, token=token)
    return updates, old_updates, info


@jax.jit
def _compute_snr_derivative(
    O_L: Array, dv: Array, grad: Array, weights, is_jac: PyTree = None, token=None, mode='complex'
):
    """
    Computes the gradient of the snr, and additional info if asked
    """
    grad_var, token = mpi.mpi_allreduce_sum_jax(O_L.T**2 @ dv**2, token=token)
    N_mc = O_L.shape[0] * mpi.n_nodes // 2
    if mode!='complex':
        raise ValueError('Automatic IS is only implemented with mode = complex')

    num_p = grad.shape[-1] // 2
    weights2 = jnp.stack([weights, weights], axis=-1)
    weights = jax.lax.collapse(weights2, 0, 2)
    dv_rw = weights * dv
    O_R, O_I = O_L.T[:, ::2], O_L.T[:, 1::2]
    dv_R, dv_I = dv[::2], dv[1::2]
    # to compute the square of the local gradient (2\Re(O_xi\Delta H_{loc}(x)^*})^2 
    # we need to compute (Re(O_xi)Re(\Delta H_{loc}(x)^*) + Im(O_xi)Im(\Delta H_{loc}(x)^*))^2, which is done below
    # We multiply by N_mc as the arrays were previously scaled in prepare_inputs by w(x)/\sqrt{N_mc}, which introduces now a factor of 1/N_mc^2
    g_loc_sq  = (O_R**2 @ dv_R**2 + O_I**2 @ dv_I**2 + 2 * (O_R*O_I) @ (dv_R*dv_I)) * N_mc

    # Then  we develop w^2(x)(f_loc^i(x) - F_i)^2 
    grad_var = (
        g_loc_sq
        - 2 * (O_L.T @ dv_rw) * grad
        + jnp.mean(weights**2) * grad ** 2
    )
    snr = jnp.abs(grad) / jnp.sqrt(grad_var)
    snr, token = distributed.allgather(snr, token=token)

    if is_jac is not None:
        is_jac2 = jax.tree_util.tree_map(
            lambda x: jax.lax.collapse(
                jnp.stack([x, x], axis=-1), 0, 2
            ),
            is_jac,
        )
        # To compute the gradient of the variance, we repeat the expression above, inserting the jacobian of the probability distribution
        grad_v = jax.tree_util.tree_map(
            lambda x,y: mpi.mpi_allreduce_sum_jax(
                (O_R**2 @ (x * dv_R**2) + O_I**2 @ (x * dv_I**2) + 2 * (O_R*O_I) @ (x*dv_R*dv_I)) * N_mc - 2 * grad * (O_L.T @ (y*dv_rw)) + (weights**2 @ y / (N_mc * 4)) * grad ** 2
            )[0], is_jac, is_jac2
        )
        
        snr_for_grad = 1 / 2 * jnp.abs(grad) / (grad_var) ** (3 / 2)
        grad_snr = jax.tree_util.tree_map(
            lambda g: jnp.mean(g * snr_for_grad, axis=-1), grad_v
        )  # (N_Pis, 2N_p) then (N_Pis,) after mean

        grad_snr = jax.tree_util.tree_map(
            lambda x: distributed.allgather(x, token=token)[0], grad_snr
        )
        return {"grad_snr": grad_snr, "snr": jnp.mean(snr)}

    else:
        return {"snr": jnp.mean(snr)}


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
