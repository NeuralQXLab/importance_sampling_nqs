import netket.jax as nkjax
import os
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp
import copy
import jax
from advanced_drivers.driver import statistics

from hydra.utils import call, instantiate, get_class, get_method
import inspect


def cumsum(lst):
    cumulative_sum = []
    total = 0
    for num in lst:
        total += num
        cumulative_sum.append(total)
    return cumulative_sum


def e_diag(H_sp):
    eig_vals, eig_vecs = eigsh(
        H_sp, k=2, which="SA"
    )  # k is the number of eigenvalues desired,
    E_gs = eig_vals[0]  # "SA" selects the ones with smallest absolute value
    return E_gs


def find_closest_saved_vals(E_err, saved_vals, save_every, n_vals_per_scale=1):
    L = len(E_err)
    exp_max = int(jnp.log10(jnp.max(E_err))) + 1
    exp_min = int(jnp.log10(jnp.min(E_err))) - 1
    exp_list = jnp.flip(jnp.arange(exp_min, exp_max))
    print((exp_max - exp_min) * n_vals_per_scale)
    exp_list = jnp.flip(
        jnp.linspace(exp_min, exp_max, (exp_max - exp_min + 1) * n_vals_per_scale)
    )
    print(exp_list)
    target_values = 10.0 ** (exp_list)  # 10^n values
    print(target_values)
    closest_saved_vals = []
    error_val = []
    for target in target_values:
        # Find the index in E_err with the value closest to the target
        closest_index = jnp.abs(E_err - target).argmin()

        # Find the corresponding index in saved_vals
        # `saved_vals` is of length L//10 and corresponds to every 10th iteration
        saved_index = closest_index // save_every
        saved_index = min(
            saved_index, len(saved_vals) - 1
        )  # Ensure index is within bounds
        if saved_vals[saved_index] not in closest_saved_vals:
            closest_saved_vals.append(saved_vals[saved_index])
            error_val.append(E_err[saved_index * save_every])
    return closest_saved_vals, error_val


def smart_instantiate(cfg, available_vars: dict, mode="instantiate"):
    if not hasattr(cfg, "_target_"):
        raise ValueError("Config missing '_target_' key")
    try:
        obj = get_class(cfg._target_)
        sig = inspect.signature(obj.__init__)
        skip_args = {"self"}
        # Expected args from the signature
        expected_args = set(sig.parameters) - skip_args
        # Inject only what matches
        injected_args = {k: v for k, v in available_vars.items() if k in expected_args}
        return instantiate(cfg, **injected_args)

    except:
        obj = get_method(cfg._target_)
        sig = inspect.signature(obj)
        skip_args = set()
        # Expected args from the signature
        expected_args = set(sig.parameters) - skip_args
        # Inject only what matches
        injected_args = {k: v for k, v in available_vars.items() if k in expected_args}

        return call(cfg, **injected_args)
