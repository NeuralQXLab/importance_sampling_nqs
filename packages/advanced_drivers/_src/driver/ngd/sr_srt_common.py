from typing import Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import netket.jax as nkjax
from netket.stats import mean as distributed_mean
from netket.utils import mpi, timing
from netket.utils.types import Union, Array, PyTree

from advanced_drivers._src.driver.ngd.sr import _compute_sr_update
from advanced_drivers._src.driver.ngd.srt import _compute_srt_update


@partial(jax.jit, static_argnames=("mode",))
def _prepare_input(
    O_L,
    local_grad,
    *,
    weights: Optional[Array] = None,
    mode: str,
) -> tuple[jax.Array, jax.Array]:
    r"""
    Prepare the input for the SR/SRt solvers.

    The local eneriges and the jacobian are reshaped, centered and normalized by the number of Monte Carlo samples.
    The complex case is handled by concatenating the real and imaginary parts of the jacobian and the local energies.

    We use [Re_x1, Im_x1, Re_x2, Im_x2, ...] so that shards are contiguous, and jax can keep track of the sharding information.
    This format is applied both to the jacobian and to the vector.

    Args:
        O_L: The jacobian of the ansatz.
        local_grad: The local energies.
        weights: The weights potentially used for importance sampling.
        mode: The mode of the jacobian: `'real'` or `'complex'`.

    Returns:
        The reshaped jacobian and the reshaped local energies.
    """
    N_mc = O_L.shape[0] * mpi.n_nodes

    if weights is None:
        weights = jnp.ones(N_mc, dtype=local_grad.dtype)

    # jacobian and local_grad are centered accounting for reweighting
    local_grad = local_grad.flatten()
    de = local_grad - distributed_mean(local_grad * weights)

    # O_L = jax.tree_util.tree_map(
    #     lambda x: subtract_mean(x, axis=0), O_L * jnp.expand_dims(weights, range(len(O_L.shape))[1:])
    # )

    # include scaling by reweighting factor so that the matrix-matrix
    # and matrix-vector products yield the correct expecation values
    weights_expanded = jnp.expand_dims(weights, range(len(O_L.shape))[1:])
    O_L = (O_L - distributed_mean(O_L * weights_expanded, axis=0)) * jnp.sqrt(
        weights_expanded / N_mc
    )
    dv = 2.0 * de * jnp.sqrt(weights / N_mc)

    if mode == "complex":
        # Concatenate the real and imaginary derivatives of the ansatz
        # (#ns, 2, np) -> (#ns*2, np)
        O_L = jax.lax.collapse(O_L, 0, 2)

        # (#ns, 2) -> (#ns*2)
        dv2 = jnp.stack([jnp.real(dv), jnp.imag(dv)], axis=-1)
        dv = jax.lax.collapse(dv2, 0, 2)
    elif mode == "real":
        dv = dv.real
    else:
        raise NotImplementedError()
    return O_L, dv


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "log_psi",
        "solver_fn",
        "mode",
        "chunk_size",
        "collect_quadratic_model",
        "collect_gradient_statistics",
        "use_ntk",
    ),
)
def _sr_srt_common(
    log_psi,
    local_grad,
    parameters,
    model_state,
    samples,
    *,
    weights: Optional[Array] = None,
    diag_shift: Union[float, Array],
    solver_fn: Callable[[Array, Array], Array],
    mode: str,
    proj_reg: Optional[Union[float, Array]] = None,
    momentum: Optional[Union[float, Array]] = None,
    old_updates: Optional[PyTree] = None,
    chunk_size: Optional[int] = None,
    collect_quadratic_model: bool = False,
    collect_gradient_statistics: bool = False,
    use_ntk: bool = False,
    is_jac: Optional[PyTree] = None,
):
    r"""
    Compute the Natural gradient update for the model specified by `log_psi({parameters, model_state}, samples)`
    and the local gradient contributions `local_grad`.

    Uses a code equivalent to QGTJacobianDense by default, or with the NTK/MinSR if `use_ntk` is True.

    Args:
        log_psi: The log of the wavefunction.
        local_grad: The local values of the estimator.
        parameters: The parameters of the model.
        model_state: The state of the model.
        samples: The samples used to compute expectation values.
        weights: The weights potentially used for importance sampling.
        diag_shift: The diagonal shift of the stochastic reconfiguration matrix. Typical values are 1e-4 ÷ 1e-3. Can also be an optax schedule.
        proj_reg: Weight before the matrix `1/N_samples \\bm{1} \\bm{1}^T` used to regularize the linear solver in SPRING.
        momentum: Momentum used to accumulate updates in SPRING.
        linear_solver_fn: Callable to solve the linear problem associated to the updates of the parameters.
        mode: The mode used to compute the jacobian of the variational state. Can be `'real'` or `'complex'` (defaults to the dtype of the output of the model).
        collect_quadratic_model: Whether to collect the quadratic model. The quantities collected are the linear and quadratic term in the approximation of the loss function. They are stored in the info dictionary of the driver.
        collect_gradient_statistics: Whether to collect the statistics (mean and variance) of the gradient. They are stored in the info dictionary of the driver.

    Returns:
        The new parameters, the old updates, and the info dictionary.
    """
    _, unravel_params_fn = ravel_pytree(parameters)
    _params_structure = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), parameters
    )

    jacobians = nkjax.jacobian(
        log_psi,
        parameters,
        samples,
        model_state,
        mode=mode,
        dense=True,
        center=False,
        chunk_size=chunk_size,
    )  # jacobian is NOT centered

    O_L, dv = _prepare_input(jacobians, local_grad, weights=weights, mode=mode)

    if old_updates is None and momentum is not None:
        old_updates = jnp.zeros(jacobians.shape[-1], dtype=jacobians.dtype)

    compute_update = _compute_srt_update if use_ntk else _compute_sr_update

    # TODO: Add support for proj_reg and momentum
    # At the moment SR does not support momentum, proj_reg.
    # We raise an error if they are passed with a value different from None.
    updates, old_updates, info = compute_update(
        O_L,
        dv,
        diag_shift=diag_shift,
        solver_fn=solver_fn,
        mode=mode,
        proj_reg=proj_reg,
        momentum=momentum,
        old_updates=old_updates,
        collect_quadratic_model=collect_quadratic_model,
        collect_gradient_statistics=collect_gradient_statistics,
        params_structure=_params_structure,
        weights=weights,
        is_jac=is_jac,
    )

    return unravel_params_fn(updates), old_updates, info


sr = partial(_sr_srt_common, use_ntk=False)

srt = partial(_sr_srt_common, use_ntk=True)


@jax.jit
def compute_snr_derivative(
    is_jac: PyTree, O_L: Array, dv: Array, grad: Array, token=None
):
    """
    Computes the gradient of the snr, and additional info if asked

    """
    raise ValueError("wrong function")
    # force_unrolled = O_L.T * dv
    # num_p = force_unrolled.shape[0]//2
    # num_s = force_unrolled.shape[1]//2

    # force_unrolled =  num_s * (force_unrolled[:, ::2] + force_unrolled[:, 1::2]) # concat along sample dim (2N_p, N_s)
    # F = jnp.mean(force_unrolled, axis=1) # force in real parametrization, (2N_p, )
    # loc_var = jnp.abs(force_unrolled - F[:, None])**2 # (2N_p, N_s)

    # loc_var_mean = jnp.mean(loc_var, axis=-1)
    # snr = jnp.abs(F)/jnp.sqrt(loc_var_mean)
    # grad_v = jax.tree_util.tree_map(
    #     lambda x: x.T @ loc_var.T,
    #     is_jac)

    grad_var, token = mpi.mpi_allreduce_sum_jax(O_L.T**2 @ dv**2, token=token)
    N_mc = O_L.shape[0] * mpi.n_nodes
    num_p = grad.shape[-1] // 2
    grad_var = grad_var * N_mc - grad**2
    grad_var = (O_L.T**2) @ (dv**2) * N_mc - (O_L.T @ dv) ** 2
    grad_v = jax.tree_util.tree_map(
        lambda x: mpi.mpi_allreduce_sum_jax(
            (O_L.T**2) @ (x * dv**2) * N_mc - 2 * grad * (O_L.T @ (x * dv))
        )[0],
        is_jac,
    )

    snr = jnp.abs(grad) / jnp.sqrt(grad_var)
    snr_for_grad = 1 / 2 * jnp.abs(grad) / (grad_var) ** (3 / 2)
    grad_snr = jax.tree_util.tree_map(
        lambda g: jnp.mean(g * snr_for_grad, axis=-1), grad_v
    )  # (N_Pis, 2N_p) then (N_Pis,) after mean
    # snr = distributed.allgather(grad,token=token)
    # grad_snr = distributed.allgather(grad,token=token)
    return {"grad_snr": grad_snr, "snr": jnp.mean(snr)}
