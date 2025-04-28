import jax.numpy as jnp
import flax

from netket import jax as nkjax


def make_logpsi_diff_afun(
    logpsi_fun, logpsi_target_fun, variables, variables_target, rescaling=None
):
    # wrap apply_fun into logpsi logpsi_U
    logpsi_diff_fun = nkjax.HashablePartial(
        _logpsi_diff_fun, logpsi_fun, logpsi_target_fun
    )

    # Insert a new 'model_state' key to store the Unitary. This only works
    # if U is a pytree that can be flattened/unflattened.
    new_variables = flax.core.copy(
        variables,
        {"variables_target": variables_target, "rescaling": rescaling},
    )

    return logpsi_diff_fun, new_variables


def _logpsi_diff_fun(afun, target_afun, variables, x):
    variables, variables_target = flax.core.pop(variables, "variables_target")
    variables_afun, rescaling = flax.core.pop(variables, "rescaling")

    if rescaling is None:
        rescaling = 1

    logpsi1_x = afun(variables_afun, x)
    logpsi2_x = target_afun(variables_target, x)

    return 0.5 * jnp.log(
        jnp.absolute(jnp.exp(logpsi1_x) - jnp.exp(logpsi2_x) / rescaling)
    )


def make_logpsi_sum_afun(
    logpsi_fun, logpsi_target_fun, variables, variables_target, epsilon
):
    # wrap apply_fun into logpsi logpsi_U
    logpsi_sum_fun = nkjax.HashablePartial(
        _logpsi_sum_fun, logpsi_fun, logpsi_target_fun
    )

    # Insert a new 'model_state' key to store the Unitary. This only works
    # if U is a pytree that can be flattened/unflattened.
    new_variables = flax.core.copy(
        variables,
        {"variables_target": variables_target, "epsilon": epsilon},
    )

    return logpsi_sum_fun, new_variables


def _logpsi_sum_fun(afun, target_afun, variables, x):
    variables, variables_target = flax.core.pop(variables, "variables_target")
    variables_afun, epsilon = flax.core.pop(variables, "epsilon")

    if epsilon is None:
        epsilon = 1

    logpsi1_x = afun(variables_afun, x)
    logpsi2_x = target_afun(variables_target, x)

    # jax.debug.print("logpsi1_x are {}", logpsi1_x)
    # jax.debug.print("logpsi2_x are {}", logpsi2_x)

    return jnp.log(
        (1 - epsilon) * jnp.exp(logpsi1_x) + epsilon * jnp.exp(logpsi2_x)
    )  ## check!
