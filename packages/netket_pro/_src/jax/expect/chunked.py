# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The score function (REINFORCE) gradient estimator of an expectation

from functools import partial

import jax
from jax import numpy as jnp

from netket.stats import statistics as mpi_statistics
from netket.utils import mpi
from netket.jax import eval_shape

from netket_pro._src.jax._vjp_chunked import vjp_chunked
from netket_pro._src.jax._vmap_chunked import apply_chunked


# log_prob_args and integrand_args are independent of params when taking the
# gradient. They can be continuous or discrete, and they can be pytrees
# Does not support higher-order derivatives yet
@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def expect_chunked(n_chains, chunk_size, log_pdf, expected_fun, pars, *σ):
    in_axes = (None,) + tuple(0 if s is not None else None for s in σ)

    L_σ = apply_chunked(
        expected_fun,
        chunk_size=chunk_size,
        in_axes=in_axes,
    )(pars, *σ)

    L̄_σ = mpi_statistics(L_σ.reshape((n_chains, -1)) if n_chains else L_σ)

    return L̄_σ.mean, L̄_σ


def expect_chunked_fwd(n_chains, chunk_size, log_pdf, expected_fun, pars, *σ):
    in_axes = (None,) + tuple(0 if s is not None else None for s in σ)
    chunk_argnums = tuple(i for i, v in enumerate(in_axes) if v is not None)

    pb_L = vjp_chunked(
        expected_fun,
        pars,
        *σ,
        chunk_size=chunk_size,
        chunk_argnums=chunk_argnums,
        return_forward=True,
    )

    # Out vector shape for backward v in vjp. Everyhting is linear so we can multiply by it later.
    out_shape = eval_shape(expected_fun, pars, *σ)
    N_samples = len(out_shape) * mpi.n_nodes
    vec = jnp.ones_like(out_shape) / N_samples

    L_σ, gradL_σ = pb_L(vec)

    L̄_stat = mpi_statistics(L_σ.reshape((n_chains, -1)) if n_chains else L_σ)
    ΔL_σ = L_σ - L̄_stat.mean

    return (L̄_stat.mean, L̄_stat), (pars, σ, gradL_σ, ΔL_σ, vec)


# TODO: in principle, the gradient of an expectation is another expectation,
# so it should support higher-order derivatives
# But I don't know how to transform log_prob_fun into grad(log_prob_fun) while
# keeping the chunk dimension and without a loop through the chunk dimension
def expect_chunked_bwd(n_chains, chunk_size, log_pdf, expected_fun, residuals, dout):
    pars, σ, gradL_σ, ΔL_σ, vec = residuals
    dL̄, dL̄_stats = dout

    in_axes = (
        None,
        0,
    ) + tuple(0 if s is not None else None for s in σ)
    chunk_argnums = tuple(i for i, v in enumerate(in_axes) if v is not None)

    def term1_fun(pars, ΔL_σ, *σ):
        log_p = log_pdf(pars, *σ)
        term1 = jax.vmap(jnp.multiply)(ΔL_σ, log_p)
        return term1

    # capture ΔL_σ to not differentiate through it
    pb_term1 = vjp_chunked(
        term1_fun,
        pars,
        ΔL_σ,
        *σ,
        chunk_size=chunk_size,
        chunk_argnums=chunk_argnums,
        nondiff_argnums=(1,),
        return_forward=False,
    )

    grad_term1 = pb_term1(vec)

    def sum_and_mpi_mean(x, y):
        # This function is equivalent to
        # mpi.mpi_mean_jax(dL̄ * (x+y))[0]
        # but does not do the operations if the x or y
        # has a float0 dtype, which means the gradient is
        # identically 0.
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.float0):
            xpy = y
        elif jax.dtypes.issubdtype(y.dtype, jax.dtypes.float0):
            xpy = x
        else:
            xpy = x + y

        if jax.dtypes.issubdtype(xpy.dtype, jax.dtypes.float0):
            return xpy
        else:
            return mpi.mpi_mean_jax(dL̄ * xpy)[0]

    grad_f = jax.tree_util.tree_map(sum_and_mpi_mean, grad_term1, gradL_σ)

    return grad_f


expect_chunked.defvjp(expect_chunked_fwd, expect_chunked_bwd)
