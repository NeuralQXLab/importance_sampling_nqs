import jax.numpy as jnp
from jax import lax


def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series
    Args:
        x: The series as a 1-D numpy array.
    Returns:
        array: The autocorrelation function of the time series.
    """
    x = jnp.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))
    # Compute the FFT and then (from that) the auto-correlation function
    f = jnp.fft.fft(x - jnp.mean(x), n=2 * n)
    acf = jnp.fft.ifft(f * jnp.conjugate(f))[: len(x)].real
    acf /= acf[0]
    return acf


def auto_window(taus, c):
    m = jnp.arange(len(taus)) < c * taus
    return lax.cond(jnp.any(m), lambda: jnp.argmin(m), lambda: len(taus) - 1)


def integrated_time(x, c=5):
    if x.ndim != 1:
        raise ValueError("invalid shape")
    print("x", x.shape, x.dtype)
    f = autocorr_1d(x)
    print("f", f.shape, f.dtype)
    taus = 2.0 * jnp.cumsum(f) - 1.0
    print("taus", taus.shape, taus.dtype)
    print("c", c)
    c = jnp.array(c, dtype=jnp.int32)
    window = auto_window(taus, c).astype(jnp.int32)
    print("auto_window", window.shape, window.dtype)
    res = taus[window]
    print("res", res.shape, res.dtype)
    return res

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

import math

import jax
import numpy as np
from jax import numpy as jnp

from netket.utils import config, mpi, struct
from netket.jax.sharding import extract_replicated
from functools import partial
import netket.jax as nkjax
# from . import mean as _mean
# from . import var as _var
# from . import total_size as _total_size


def _mean(a, w, axis=None, keepdims: bool = False):
    arr = a * w
    out = arr.mean(axis=axis, keepdims=keepdims)
    out, _ = mpi.mpi_mean_jax(out)
    return out


def sum(a, axis=None, keepdims: bool = False):
    # if it's a numpy-like array...
    if hasattr(a, "shape"):
        # use jax
        a_sum = a.sum(axis=axis, keepdims=keepdims)
    else:
        # assume it's a scalar
        a_sum = jnp.asarray(a)
    out, _ = mpi.mpi_sum_jax(a_sum)
    return out


def _var(a, w, axis=None, ddof: int = 0):
    m = _mean(a, w, axis=axis)
    if axis is None:
        ssq = (w ** 2.0) * jnp.abs(a - m) ** 2.0
    else:
        ssq = (w ** 2.0) * jnp.abs(a - jnp.expand_dims(m, axis)) ** 2.0
    out = sum(ssq, axis=axis)
    n_all = _total_size(a, axis=axis)
    out /= n_all - ddof
    return out


def _total_size(a, axis=None):
    if axis is None:
        l_size = a.size
    else:
        l_size = a.shape[axis]
    return l_size * mpi.n_nodes

def _format_decimal(value, std, var):
    if math.isfinite(std) and std > 1e-7:
        decimals = max(int(np.ceil(-np.log10(std))), 0)
        return (
            "{0:.{1}f}".format(value, decimals + 1),
            "{0:.{1}f}".format(std, decimals + 1),
            "{0:.{1}f}".format(var, decimals + 1),
        )
    else:
        return (
            f"{value:.3e}",
            f"{std:.3e}",
            f"{var:.3e}",
        )


_NaN = float("NaN")


def _maybe_item(x):
    if hasattr(x, "shape") and x.shape == ():
        return x.item()
    else:
        return x


@struct.dataclass
class Stats:
    """A dict-compatible pytree containing the result of the statistics function."""

    mean: float | complex = _NaN
    """The mean value."""
    error_of_mean: float = _NaN
    """Estimate of the error of the mean."""
    variance: float = _NaN
    """Estimation of the variance of the data."""
    tau_corr: float = _NaN
    """Estimate of the autocorrelation time (in dimensionless units of number of steps).

    This value is estimated with a blocking algorithm by default, but the result is known
    to be unreliable. A more precise estimator based on the FFT transform can be used by
    setting the environment variable `NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION=1`. This
    estimator is more computationally expensive, but overall the added cost should be
    negligible.
    """
    R_hat: float = _NaN
    """
    Estimator of the split-Rhat convergence estimator.

    The split-Rhat diagnostic is based on comparing intra-chain and inter-chain
    statistics of the sample and is thus only available for 2d-array inputs where
    the rows are independently sampled MCMC chains. In an ideal MCMC samples,
    R_hat should be 1.0. If it deviates from this value too much, this indicates
    MCMC convergence issues. Thresholds such as R_hat > 1.1 or even R_hat > 1.01 have
    been suggested in the literature for when to discard a sample. (See, e.g.,
    Gelman et al., `Bayesian Data Analysis <http://www.stat.columbia.edu/~gelman/book/>`_,
    or Vehtari et al., `arXiv:1903.08008 <https://arxiv.org/abs/1903.08008>`_.)
    """
    tau_corr_max: float = _NaN
    """
    Estimate of the maximum autocorrelation time among all Markov chains.

    This value is only computed if the environment variable
    `NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION` is set.
    """

    def to_dict(self):
        jsd = {}
        jsd["Mean"] = _maybe_item(self.mean)
        jsd["Variance"] = _maybe_item(self.variance)
        jsd["Sigma"] = _maybe_item(self.error_of_mean)
        jsd["R_hat"] = _maybe_item(self.R_hat)
        jsd["TauCorr"] = _maybe_item(self.tau_corr)
        # if config.netket_experimental_fft_autocorrelation:
        jsd["TauCorrMax"] = _maybe_item(self.tau_corr_max)
        return jsd

    def to_compound(self):
        return "Mean", self.to_dict()

    def __repr__(self):
        # extract adressable data from fully replicated arrays
        self = extract_replicated(self)
        mean, err, var = _format_decimal(self.mean, self.error_of_mean, self.variance)
        if not math.isnan(self.R_hat):
            ext = f", R̂={self.R_hat:.4f}"
        else:
            ext = ""
        if config.netket_experimental_fft_autocorrelation:
            if not (math.isnan(self.tau_corr) and math.isnan(self.tau_corr_max)):
                ext += f", τ={self.tau_corr:.1f}<{self.tau_corr_max:.1f}"
        return f"{mean} ± {err} [σ²={var}{ext}]"

    # Alias accessors
    def __getattr__(self, name):
        if name in ("mean", "Mean"):
            return self.mean
        elif name in ("variance", "Variance"):
            return self.variance
        elif name in ("error_of_mean", "Sigma"):
            return self.error_of_mean
        elif name in ("R_hat", "R"):
            return self.R_hat
        elif name in ("tau_corr", "TauCorr"):
            return self.tau_corr
        elif name in ("tau_corr_max", "TauCorrMax"):
            return self.tau_corr_max
        else:
            raise AttributeError(f"'Stats' object object has no attribute '{name}'")

    def real(self):
        return self.replace(mean=np.real(self.mean))

    def imag(self):
        return self.replace(mean=np.imag(self.mean))


def _get_blocks(data, block_size):
    chain_length = data.shape[1]

    n_blocks = int(np.floor(chain_length / float(block_size)))

    return data[:, 0 : n_blocks * block_size].reshape((-1, block_size)).mean(axis=1)


def _block_variance(data, weights, l):
    blocks = _get_blocks(data, l)
    blocks_w = _get_blocks(weights, l)
    print(blocks.shape)
    ts = _total_size(blocks)
    print(ts)
    if ts > 0:
        return _var(blocks, blocks_w), ts
    else:
        return jnp.nan, 0


def _batch_variance(data, weights):
    b_means = data.mean(axis=1)
    w_means = weights.mean(axis=1)
    ts = _total_size(b_means)
    return _var(b_means, w_means), ts


# this is not batch_size maybe?
def statistics(data, weights):
    r"""
    Returns statistics of a given array (or matrix, see below) containing a stream of data.
    This is particularly useful to analyze Markov Chain data, but it can be used
    also for other type of time series.
    Assumes same shape on all MPI processes.

    Args:
        data (vector or matrix): The input data. It can be real or complex valued.
            * if a vector, it is assumed that this is a time series of data (not necessarily independent);
            * if a matrix, it is assumed that that rows :code:`data[i]` contain independent time series.

    Returns:
       Stats: A dictionary-compatible class containing the
             average (:code:`.mean`, :code:`["Mean"]`),
             variance (:code:`.variance`, :code:`["Variance"]`),
             the Monte Carlo standard error of the mean (:code:`error_of_mean`, :code:`["Sigma"]`),
             an estimate of the autocorrelation time (:code:`tau_corr`, :code:`["TauCorr"]`), and the
             Gelman-Rubin split-Rhat diagnostic (:code:`.R_hat`, :code:`["R_hat"]`).

             These properties can be accessed both the attribute and the dictionary-style syntax
             (both indicated above).

             The split-Rhat diagnostic is based on comparing intra-chain and inter-chain
             statistics of the sample and is thus only available for 2d-array inputs where
             the rows are independently sampled MCMC chains. In an ideal MCMC samples,
             R_hat should be 1.0. If it deviates from this value too much, this indicates
             MCMC convergence issues. Thresholds such as R_hat > 1.1 or even R_hat > 1.01 have
             been suggested in the literature for when to discard a sample. (See, e.g.,
             Gelman et al., `Bayesian Data Analysis <http://www.stat.columbia.edu/~gelman/book/>`_,
             or Vehtari et al., `arXiv:1903.08008 <https://arxiv.org/abs/1903.08008>`_.)
    """
    return _statistics(data, weights)


# @partial(jax.jit, static_argnums=2)
# def _statistics(data, weights, batch_size):
#     data = jnp.atleast_1d(data)
#     if data.ndim == 1:
#         data = data.reshape((1, -1))
#         weights = weights.reshape((1, -1))

#     if data.ndim > 2:
#         raise NotImplementedError("Statistics are implemented only for ndim<=2")

#     mean = _mean(data, weights)
#     variance = _var(data, weights)
#     jax.debug.print('x={x}', x=variance)
#     ts = _total_size(data)

#     bare_var = variance

#     batch_var, n_batches = _batch_variance(data, weights)

#     l_block = max(1, data.shape[1] // batch_size)
#     print(l_block)
#     block_var, n_blocks = _block_variance(data, weights, l_block)

#     jax.debug.print('blockvar {x}' , x = block_var)
#     jax.debug.print('batchvar {x}' , x = batch_var)

#     tau_batch = ((ts / n_batches) * batch_var / bare_var - 1) * 0.5
#     tau_block = ((ts / n_blocks) * block_var / bare_var - 1) * 0.5

#     batch_good = (tau_batch < 6 * data.shape[1]) * (n_batches >= batch_size)
#     block_good = (tau_block < 6 * l_block) * (n_blocks >= batch_size)
#     print(batch_good, block_good)
#     stat_dtype = nkjax.dtype_real(data.dtype)

#     # if batch_good:
#     #    error_of_mean = jnp.sqrt(batch_var / n_batches)
#     #    tau_corr = jnp.max(0, tau_batch)
#     # elif block_good:
#     #    error_of_mean = jnp.sqrt(block_var / n_blocks)
#     #    tau_corr = jnp.max(0, tau_block)
#     # else:
#     #    error_of_mean = jnp.nan
#     #    tau_corr = jnp.nan
#     # jax style

#     def batch_good_err(args):
#         batch_var, tau_batch, *_ = args
#         error_of_mean = jnp.sqrt(batch_var / n_batches)
#         tau_corr = jnp.clip(tau_batch, 0)
#         return jnp.asarray(error_of_mean, dtype=stat_dtype), jnp.asarray(
#             tau_corr, dtype=stat_dtype
#         )

#     def block_good_err(args):
#         _, _, block_var, tau_block = args
#         error_of_mean = jnp.sqrt(block_var / n_blocks)
#         tau_corr = jnp.clip(tau_block, 0)
#         return jnp.asarray(error_of_mean, dtype=stat_dtype), jnp.asarray(
#             tau_corr, dtype=stat_dtype
#         )

#     def nan_err(args):
#         return jnp.asarray(jnp.nan, dtype=stat_dtype), jnp.asarray(
#             jnp.nan, dtype=stat_dtype
#         )

#     def batch_not_good(args):
#         batch_var, tau_batch, block_var, tau_block, block_good = args
#         return jax.lax.cond(
#             block_good,
#             block_good_err,
#             nan_err,
#             (batch_var, tau_batch, block_var, tau_block),
#         )

#     error_of_mean, tau_corr = jax.lax.cond(
#         batch_good,
#         batch_good_err,
#         batch_not_good,
#         (batch_var, tau_batch, block_var, tau_block, block_good),
#     )

#     if n_batches > 1:
#         N = data.shape[-1]

#         if not config.netket_use_plain_rhat:
#             # compute split-chain batch variance
#             local_batch_size = data.shape[0]
#             if N % 2 == 0:
#                 # split each chain in the middle,
#                 # like [[1 2 3 4]] -> [[1 2][3 4]]
#                 batch_var, _ = _batch_variance(
#                     data.reshape(2 * local_batch_size, N // 2), weights.reshape(2 * local_batch_size, N // 2)
#                 )
#             else:
#                 # drop the last sample of each chain for an even split,
#                 # like [[1 2 3 4 5]] -> [[1 2][3 4]]
#                 batch_var, _ = _batch_variance(
#                     data[:, :-1].reshape(2 * local_batch_size, N // 2), weights[:, :-1].reshape(2 * local_batch_size, N // 2)
#                 )

#         # V_loc = _np.var(data, axis=-1, ddof=0)
#         # W_loc = _np.mean(V_loc)
#         # W = _mean(W_loc)
#         # # This approximation seems to hold well enough for larger n_samples
#         W = variance

#         R_hat = jnp.sqrt((N - 1) / N + batch_var / W)
#     else:
#         R_hat = jnp.nan

#     res = Stats(mean, error_of_mean, variance, tau_corr, R_hat)

#     return res

def _split_R_hat(data, W):
    N = data.shape[-1]
    # if not config.netket_use_plain_rhat:
    # compute split-chain batch variance
    local_batch_size = data.shape[0]
    if N % 2 == 0:
        # split each chain in the middle,
        # like [[1 2 3 4]] -> [[1 2][3 4]]
        batch_var, _ = _batch_variance(data.reshape(2 * local_batch_size, N // 2))
    else:
        # drop the last sample of each chain for an even split,
        # like [[1 2 3 4 5]] -> [[1 2][3 4]]
        batch_var, _ = _batch_variance(
            data[:, :-1].reshape(2 * local_batch_size, N // 2)
        )

    # V_loc = _np.var(data, axis=-1, ddof=0)
    # W_loc = _np.mean(V_loc)
    # W = _mean(W_loc)
    # # This approximation seems to hold well enough for larger n_samples
    return jnp.sqrt((N - 1) / N + batch_var / W)


BLOCK_SIZE = 32
       
#@jax.jit
def _statistics(data, weights):
    print("data", data.shape, data.dtype, data.sharding)
    print("weights", weights.shape, weights.dtype, weights.sharding)
    data = jnp.atleast_1d(data)
    if data.ndim == 1:
        data = data.reshape((1, -1))
        weights = weights.reshape((1, -1))

    if data.ndim > 2:
        raise NotImplementedError("Statistics are implemented only for ndim<=2")

    mean = _mean(data, weights)
    variance = _var(data, weights)

    taus = jax.vmap(integrated_time)(data*weights)
    tau_avg, _ = mpi.mpi_mean_jax(jnp.mean(taus))
    tau_max, _ = mpi.mpi_max_jax(jnp.max(taus))

    batch_var, n_batches = _batch_variance(data, weights)
    if n_batches > 1:
        error_of_mean = jnp.sqrt(batch_var / n_batches)
        R_hat = _split_R_hat(data*weights, variance)
    else:
        l_block = max(1, data.shape[1] // BLOCK_SIZE)
        block_var, n_blocks = _block_variance(data, weights, l_block)
        error_of_mean = jnp.sqrt(block_var / n_blocks)
        R_hat = jnp.nan
    error_of_mean = variance * jnp.sqrt(tau_avg)
    res = Stats(mean, error_of_mean, variance, tau_avg, R_hat, tau_max)

    return res