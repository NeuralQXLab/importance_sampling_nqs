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

from typing import Callable, Optional

from jax import numpy as jnp

from netket.stats import Stats
from netket.utils.types import PyTree

from netket_pro._src.jax.expect.standard import expect as expect_nonchunked
from netket_pro._src.jax.expect.chunked import expect_chunked
from netket_pro._src.jax.expect.full_sum import expect_fullsum


def expect_advanced(
    log_pdf: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    expected_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    pars: PyTree,
    *σ: tuple[jnp.ndarray, ...],
    n_chains: Optional[int] = None,
    chunk_size: Optional[int] = None,
    full_sum: bool = False,
) -> tuple[jnp.ndarray, Stats]:
    r"""
    A special version of nk.jax.expect, which:
     - Supports chunking and sharding
     - Is a x2 factor
     - Does not support (yet) extra arguments to the expect function


    Computes the expectation value over a log-pdf, equivalent to

    .. math::

        \langle f \rangle = \mathbb{E}_{\sigma \sim p(x)}[f(\sigma)] = \frac{\sum_{\mathbf{x}} p(\mathbf{x}) f(\mathbf{x})}{\sum_{\mathbf{x}} p(\mathbf{x})}

    where the evaluation of the expectation value is approximated using the sample average, with
    samples :math:`\sigma` that are assumed to be drawn from the non-normalized probability distribution :math:`p(x)`.

    .. math::

        \langle f \rangle \approx \frac{1}{N} \sum_{i=1}^{N} f(\sigma_i)

    This function ensures that the backward pass is computed correctly, by first differentiating the first equation
    above, and then by approximating the expectation values again using the sample average. The resulting
    backward gradient is

    .. math::

            \nabla \langle f \rangle = \mathbb{E}_{\sigma \sim p(x)}[(\nabla \log p(\sigma) - \langle\log p(\sigma)\rangle) f(\sigma) + \nabla f(\sigma)]

    where again, the expectation values are comptued using the sample average. The centering term arises from taking the derivative
    of the denominator.

    .. warning::

        Full Sum mode does not support MPI or chunking.

    Args:
        log_pdf: The log-pdf function from which the samples are drawn. This should output real values, and have a signature
            :code:`log_pdf(pars, σ) -> jnp.ndarray`.
        expected_fun: The function to compute the expectation value of. This should have a signature
            :code:`expected_fun(pars, σ, *expected_fun_args) -> jnp.ndarray`.
        pars: The parameters of the model.
        σ: The samples to compute the expectation value over.
        n_chains: The number of chains to use in the computation. If None, the number of chains is inferred from the shape of the input.
        chunk_size: The size of the chunks to use in the computation. If None, no chunking is used.
        full_sum: If True, assumes that the configurations span the full configuration space and returns the exact
            expectation value and gradient (defaults to False).

    Returns:
        A tuple where the first element is the scalar value containing the expectation value, and the second element is
        a :class:`netket.stats.Stats` object containing the statistics (including the mean) of the expectation value.

    .. note::

        When using this function together with MPI, you have to pay particular attention. This is because inside the function `f` that is differentiated
        a mean over the MPI ranks (`mpi_mean(term1 + term2, axis=0)`) appears. Therefore, when doing the backward pass this results in a division of the outputs
        from the previous steps by a factor equal to the number of MPI ranks, and so the final gradient on each MPI rank is rescaled as well.
        To cope with this, it is important to sum over the ranks the gradient computed after AD, for example using the function `nk.utils.mpi.mpi_sum_jax`.
        See the following example for more details.

    Example:
        Compute the energy gradient using `nk.jax.expect` on more MPI ranks.

        >>> import netket as nk
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> hi = nk.hilbert.Spin(s=0.5, N=20)
        >>> graph = nk.graph.Chain(length=20)
        >>> H = nk.operator.IsingJax(hi, graph, h=1.0)
        >>> vstate = nk.vqs.MCState(sampler=nk.sampler.MetropolisLocal(hi, n_chains_per_rank=16), model=nk.models.RBM(alpha=1, param_dtype=complex), n_samples=100000)
        >>>
        >>> afun = vstate._apply_fun
        >>> pars = vstate.parameters
        >>> model_state = vstate.model_state
        >>> log_pdf = lambda params, σ: 2 * afun({"params": params, **model_state}, σ).real
        >>>
        >>> σ = vstate.samples
        >>> σ = σ.reshape(-1, σ.shape[-1])
        >>>
        >>> # The function that we want to differentiate wrt pars and σ
        >>> # Note that we do not want to compute the gradient wrt model_state, so
        >>> # we capture it inside of this function.
        >>> def expect(pars, σ):
        ...
        ...     # The log probability distribution we have generated samples σ from.
        ...     def log_pdf(pars, σ):
        ...         W = {"params": pars, **model_state}
        ...         return 2 * afun(W, σ).real
        ...
        ...     def expected_fun(pars, σ):
        ...         W = {"params": pars, **model_state}
        ...         # Get connected samples
        ...         σp, mels = H.get_conn_padded(σ)
        ...         logpsi_σ = afun(W, σ)
        ...         logpsi_σp = afun(W, σp)
        ...         logHpsi_σ = jax.scipy.special.logsumexp(logpsi_σp, b=mels, axis=1)
        ...         return jnp.exp(logHpsi_σ - logpsi_σ)
        ...     return nk.jax.expect(log_pdf, expected_fun, pars, σ)[0]
        >>>
        >>> E, E_vjp_fun = nk.jax.vjp(expect, pars, σ)
        >>> grad = E_vjp_fun(jnp.ones_like(E))[0]
        >>> grad = jax.tree_util.tree_map(lambda x: nk.utils.mpi.mpi_sum_jax(x)[0], grad)


    """
    if chunk_size is not None:
        if σ[0].shape[0] <= chunk_size:
            chunk_size = None

    if full_sum:
        return expect_fullsum(log_pdf, expected_fun, pars, *σ)
    elif chunk_size is None:
        return expect_nonchunked(n_chains, log_pdf, expected_fun, pars, *σ)
    else:
        return expect_chunked(
            n_chains,
            chunk_size,
            log_pdf,
            expected_fun,
            pars,
            *σ,
        )
