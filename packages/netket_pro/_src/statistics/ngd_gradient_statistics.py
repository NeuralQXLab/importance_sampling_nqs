import jax
import jax.numpy as jnp
from flax import core as fcore

from netket.jax import jacobian, tree_size

from advanced_drivers._src.driver.ngd.sr_srt_common import _prepare_input
from advanced_drivers._src.driver.ngd.driver_abstract_ngd import _flatten_samples
from advanced_drivers._src.driver.ngd.infidelity_kernels import cmc_kernel, smc_kernel

from functools import partial
from netket.utils.types import Array
from typing import Optional


@partial(jax.jit, static_argnames=("afun", "afun_t", "estimator"))
def compute_stats_of_covgradient(
    afun, vars, samples, afun_t, vars_t, samples_t, estimator: str, cv_coeff=-0.5
):
    r"""
    TODO: Generalize to `mode` other than `"complex"`.
    """
    model_state, params = fcore.pop(vars, "params")

    samples = _flatten_samples(samples)
    samples_t = _flatten_samples(samples_t)

    estimator_fn = cmc_kernel if estimator == "cmc" else smc_kernel

    local_grad, _ = estimator_fn(
        afun,
        vars,
        samples,
        afun_t,
        vars_t,
        samples_t,
        cv_coeff=cv_coeff,
    )

    jacobians = jacobian(
        afun,
        params,
        samples,
        model_state,
        mode="complex",
        dense=True,
        center=True,
    )  # jacobian is centered

    Ns = jacobians.shape[0]
    Np = tree_size(params)

    O_L, dv = _prepare_input(
        jacobians, local_grad, mode="complex", e_mean=local_grad.mean()
    )

    mean_re_im = O_L.T @ dv
    var_re_im = O_L.T**2 @ dv**2 * Ns - mean_re_im**2

    mean = mean_re_im[:Np] + 1j * mean_re_im[Np:]
    var = var_re_im[:Np] + var_re_im[Np:]
    return mean, var


def compute_stats_of_covgradient_from_driver(
    driver, estimator, samples: Optional[Array] = None
):
    afun, vars, σ, extra_args = driver._get_local_estimators_kernel_args()
    afun_t, vars_t, σ_t, cv = extra_args

    if samples is not None:
        σ = samples

    return compute_stats_of_covgradient(
        afun, vars, σ, afun_t, vars_t, σ_t, estimator, cv
    )


@partial(jax.jit, static_argnames=("afun", "afun_t"))
def compute_nonhermitian_gradient_estimator_holomorphic(
    afun, vars, samples, afun_t, vars_t, samples_t, cv_coeff=-0.5
):
    r"""
    TODO: Generalize to `mode` other than `"holomorphic"`.
    """
    samples = _flatten_samples(samples)
    samples_t = _flatten_samples(samples_t)

    logψ_x = afun(vars, samples)
    logϕ_x = afun_t(vars_t, samples)
    logψ_y = afun(vars, samples_t)
    logϕ_y = afun_t(vars_t, samples_t)

    logF = logψ_y + logϕ_x - logψ_x - logϕ_y
    A = jnp.exp(logF).flatten()

    F = A.real
    c = cv_coeff if cv_coeff is not None else 0.0
    F = F + c * (jnp.abs(A) ** 2 - 1)
    F = F.flatten()

    model_state, params = fcore.pop(vars, "params")

    Jx = jacobian(
        apply_fun=afun,
        params=params,
        samples=samples,
        mode="holomorphic",
        center=False,
        dense=True,
        model_state=model_state,
    )

    Jy = jacobian(
        apply_fun=afun,
        params=params,
        samples=samples_t,
        mode="holomorphic",
        center=False,
        dense=True,
        model_state=model_state,
    )

    ΔJx = Jx - Jx.mean(0)
    grad_Aconj = A.conj()[:, None] * (Jy - Jx).conj()
    grad_F = 0.5 * grad_Aconj + c * A[:, None] * grad_Aconj

    g_nh_loc = -2 * (ΔJx.conj() * F[:, None] + grad_F)
    return g_nh_loc


def compute_stats_of_nonhermgradient(
    afun, vars, samples, afun_t, vars_t, samples_t, cv_coeff=-0.5
):
    g_nh_loc = compute_nonhermitian_gradient_estimator_holomorphic(
        afun, vars, samples, afun_t, vars_t, samples_t, cv_coeff
    )
    return g_nh_loc.mean(0), g_nh_loc.var(0)


def compute_stats_of_nonhermgradient_from_driver(
    driver, samples: Optional[Array] = None
):
    afun, vars, σ, extra_args = driver._get_local_estimators_kernel_args()
    afun_t, vars_t, σ_t, cv = extra_args

    if samples is not None:
        σ = samples

    return compute_stats_of_nonhermgradient(afun, vars, σ, afun_t, vars_t, σ_t, cv)


def compute_stats_of_infidelity_from_driver(driver):
    afun, vars, samples, extra_args = driver._get_local_estimators_kernel_args()
    afun_t, vars_t, samples_t, c = extra_args

    samples = _flatten_samples(samples)
    samples_t = _flatten_samples(samples_t)

    logψ_x = afun(vars, samples)
    logϕ_x = afun_t(vars_t, samples)
    logψ_y = afun(vars, samples_t)
    logϕ_y = afun_t(vars_t, samples_t)

    logRϕψ = logϕ_x - logψ_x
    logRψϕ = logψ_y - logϕ_y
    logA = logRϕψ + logRψϕ

    Rϕψ_x = jnp.exp(logRϕψ)
    Rψϕ_y = jnp.exp(logRψϕ)
    A_xy = jnp.exp(logA)

    E = jnp.mean(Rψϕ_y)
    E2 = jnp.mean(jnp.abs(Rψϕ_y) ** 2)

    smc = A_xy
    dmc = Rϕψ_x * E

    smc_cv = smc.real + c * (jnp.abs(A_xy) ** 2 - 1)
    dmc_cv = dmc.real + c * (jnp.abs(Rϕψ_x) ** 2 * E2 - 1)

    smc = 1 - smc
    dmc = 1 - dmc
    smc_cv = 1 - smc_cv
    dmc_cv = 1 - dmc_cv

    smc_mean = jnp.mean(smc).real
    dmc_mean = jnp.mean(dmc).real
    smc_cv_mean = jnp.mean(smc_cv).real
    dmc_cv_mean = jnp.mean(dmc_cv).real

    smc_var = jnp.var(smc)
    dmc_var = jnp.var(dmc)
    smc_cv_var = jnp.var(smc_cv)
    dmc_cv_var = jnp.var(dmc_cv)

    return (
        smc_mean,
        smc_var,
        dmc_mean,
        dmc_var,
        smc_cv_mean,
        smc_cv_var,
        dmc_cv_mean,
        dmc_cv_var,
    )
