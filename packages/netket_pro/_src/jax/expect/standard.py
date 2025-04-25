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

from netket.stats import statistics as mpi_statistics, mean as mpi_mean
from netket.jax import vjp as nkvjp


# log_prob_args and integrand_args are independent of params when taking the
# gradient. They can be continuous or discrete, and they can be pytrees
# Does not support higher-order derivatives yet
@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def expect(n_chains, log_pdf, expected_fun, pars, *σ):
    L_σ = expected_fun(pars, *σ)
    if n_chains is not None:
        L_σ = L_σ.reshape((n_chains, -1))

    L̄_σ = mpi_statistics(L_σ)
    # L̄_σ = L_σ.mean(axis=0)

    return L̄_σ.mean, L̄_σ


def expect_fwd(n_chains, log_pdf, expected_fun, pars, *σ):
    # The forward pass is in principle as easy as calling and averaging over
    # expected_fun. However, to avoid calling this function twice, in here and
    # in the backward pass, we directly build the vjp, with an auxiliary output.
    #
    # L_σ = expected_fun(pars, σ, *expected_fun_args)

    def f(pars, *σ):
        # This is the real forward pass.
        L_σ = expected_fun(pars, *σ)

        L_σ_r = L_σ.reshape((n_chains, -1)) if n_chains else L_σ

        L̄_stat = jax.lax.stop_gradient(mpi_statistics(jax.lax.stop_gradient(L_σ_r)))

        # We could use L̄_mean = L̄_stat.mean but we must stop the gradient
        # Let's hope DCE deduplicates this operation...
        L̄_mean = mpi_mean(L_σ_r)

        # We have two terms in the gradient: ∇(p⋅L) = p⋅(∇logp⋅L + ∇L)
        # Below we write a function whose gradient will lead to the gradient above
        # (Excluding the p(x) which is implicit because of sampling estimation).
        # We do not really care about the output of this function. Only of its gradient!

        # We will use the baseline trick to reduce the variance
        ΔL_σ = L_σ - L̄_mean

        # We first compute something whose gradient evaluates to the first term
        log_p = log_pdf(pars, *σ)
        log_p_L_σ = jax.vmap(jnp.multiply)(jax.lax.stop_gradient(ΔL_σ), log_p)

        # And we add to it the term whose gradient evaluates to ∇L
        out = mpi_mean(log_p_L_σ + L_σ, axis=0)
        out = out.sum()
        return out, L̄_stat

    L_σ, pb, L̄_stat = nkvjp(f, pars, *σ, has_aux=True)

    return (L̄_stat.mean, L̄_stat), (pb,)


# TODO: in principle, the gradient of an expectation is another expectation,
# so it should support higher-order derivatives
# But I don't know how to transform log_prob_fun into grad(log_prob_fun) while
# keeping the chunk dimension and without a loop through the chunk dimension
def expect_bwd(n_chains, log_pdf, expected_fun, residuals, dout):
    (pb,) = residuals
    dL̄, dL̄_stats = dout
    grad_f = pb(dL̄)
    return grad_f


expect.defvjp(expect_fwd, expect_bwd)
