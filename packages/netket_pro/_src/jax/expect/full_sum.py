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

from jax import numpy as jnp

from netket.stats import Stats


def expect_fullsum(log_pdf, expected_fun, pars, *σ):
    log_pdf_σ = log_pdf(pars, *σ)
    A_loc = expected_fun(pars, *σ)
    # Hardcode to zero the values where the pdf is exactly 0
    A_loc = jnp.where(log_pdf_σ == -jnp.inf, 0, A_loc)

    p_sigma = jnp.exp(log_pdf_σ)
    # support pdfs that are identically zero.
    Z = jnp.sum(p_sigma)
    Z = jnp.where(Z == 0, 1, Z)
    p_sigma = p_sigma / jnp.sum(p_sigma)

    A_mean = jnp.sum(A_loc * p_sigma)
    A_variance = jnp.sum((A_loc - A_mean) ** 2 * p_sigma)
    A_stats = Stats(mean=A_mean, variance=A_variance, error_of_mean=0.0)

    return A_mean, A_stats
