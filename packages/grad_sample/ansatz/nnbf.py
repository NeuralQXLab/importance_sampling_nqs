import flax.linen as nn
import netket as nk
import jax.numpy as jnp
from typing import Any
import jax

from netket.nn.masked_linear import default_kernel_init

DType = Any


class MLP(nn.Module):
    n_layers: int
    n_features: int
    n_out: int
    param_dtype: DType = float
    hidden_activation: nn.activation = nn.gelu
    out_activation: nn.activation = nn.tanh
    kernel_init: Any = default_kernel_init

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.Dense(
                self.n_features,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
            )(x)
            x = self.hidden_activation(x)
        x = nn.Dense(
            self.n_out, param_dtype=self.param_dtype, kernel_init=self.kernel_init
        )(x)
        x = self.out_activation(x)
        return x


from functools import partial
from typing import Any

import jax.numpy as jnp
import jax
import flax.linen as nn
import numpy as np

from netket.experimental.hilbert import SpinOrbitalFermions
from netket.utils.types import NNInitFunc

from netket.jax import logsumexp_cplx
from netket.utils.group import PermutationGroup

default_init_func = nn.initializers.lecun_normal()

# Old versions of jax do not have jnp.bool defined.
if not hasattr(jax.numpy, "bool"):
    jnp.bool = jnp.bool_


def _log_det(A):
    sign, logabsdet = jnp.linalg.slogdet(A)
    return logabsdet.astype(complex) + jnp.log(sign.astype(complex))


_log_det = jax.jit(_log_det)


class Backflow_noMF(nn.Module):
    """
    Backflow with a jastrow factor that is only distance-dependent (aka, translational invariant).
    """

    model: Any
    hilbert: SpinOrbitalFermions
    """
    Lattice used to compute the distance between sites for the jastrow.

    If none, it is not used.
    """
    graph: Any = None
    enforce_spin_flip: bool = False
    mean_field_init: str = "default"
    param_dtype: Any = float
    initializer: NNInitFunc = nn.initializers.he_uniform()

    def batch_spin_flip(self, n):
        """From a batch of samples, include the spin-flipped states."""

        if self.hilbert.n_fermions_per_spin[0] != self.hilbert.n_fermions_per_spin[1]:
            raise ValueError(
                "Spin sectors must have the same number of fermions in order to +\
                             enforce spin-flip symmetry."
            )

        n_flip = jnp.concatenate(
            [n[:, self.hilbert.n_orbitals :], n[:, : self.hilbert.n_orbitals]], axis=1
        )
        n = jnp.concatenate([n, n_flip], axis=0)
        return n

    def psi_eval_spin_flip(self, logpsi):
        """From a batch of log-amplitudes that include the spin-flipped states,
        return the log amplitudes projected onto the spin-flip symmetric subspace."""

        logpsi_flipped = logpsi.reshape(
            (-1, 2), order="F"
        )  # reshape to (Nbatch, Nsymm=2) [log(psi(sigma)), log(psi(Psigma))]
        logpsi_symm = logsumexp_cplx(a=logpsi_flipped, b=1 / 2, axis=1)
        return logpsi_symm

    @nn.compact
    def __call__(self, n):

        # spin flipped samples
        if self.enforce_spin_flip:
            n = self.batch_spin_flip(n)

        F = self.model(n)

        @partial(jnp.vectorize, signature="(n),(m)->()")
        def log_sdj(n, F):

            # Find the positions of the occupied orbitals
            R_u = n[: self.hilbert.n_orbitals].nonzero(
                size=self.hilbert.n_fermions_per_spin[0]
            )[0]
            R_d = n[self.hilbert.n_orbitals :].nonzero(
                size=self.hilbert.n_fermions_per_spin[1]
            )[0]

            # reshape into M and add
            M = F.reshape(self.hilbert.n_orbitals, self.hilbert.n_fermions)

            # Extract the Nf x Nf submatrix of M corresponding to the occupied orbitals
            A_u = M[:, : self.hilbert.n_fermions_per_spin[0]][R_u]
            A_d = M[:, self.hilbert.n_fermions_per_spin[0] :][R_d]

            return _log_det(A_u) + _log_det(A_d)

        log_slater = log_sdj(n, F)

        # project on spin flip subspace
        if self.enforce_spin_flip:
            log_slater = self.psi_eval_spin_flip(log_slater)

        return log_slater


class LogNeuralBackflow(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    n_layers: int
    hidden_units: int
    kernel_init: Any = default_kernel_init
    param_dtype: Any = jnp.float32

    def setup(self):
        """Initialize model parameters."""
        # The N x Nf matrix of the orbitals
        self.backflow = Backflow_noMF(
            model=MLP(
                n_layers=1,
                n_features=self.hilbert.size,
                hidden_activation=nn.gelu,
                n_out=self.hilbert.n_orbitals * self.hilbert.n_fermions,
            ),
            hilbert=self.hilbert,
            enforce_spin_flip=True,
        )

    def __call__(self, n: jax.Array) -> jax.Array:
        """Vectorized computation over batches."""
        return self.backflow(n)
