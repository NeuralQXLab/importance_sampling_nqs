import flax.linen as nn
import netket as nk
import jax.numpy as jnp
from typing import Any
import jax

from netket.nn.masked_linear import default_kernel_init
from netket_fermions.models import Backflow_noMF
 
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
            x = nn.Dense(self.n_features, param_dtype=self.param_dtype, kernel_init=self.kernel_init)(x)
            x = self.hidden_activation(x)
        x = nn.Dense(self.n_out, param_dtype=self.param_dtype, kernel_init=self.kernel_init)(x)
        x = self.out_activation(x)
        return x

class LogNeuralBackflow(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    n_layers: int
    hidden_units: int
    kernel_init: Any = default_kernel_init
    param_dtype: Any = jnp.float32

    def setup(self):
        """Initialize model parameters."""
        # The N x Nf matrix of the orbitals
        self.backflow = Backflow_noMF(model = MLP(n_layers=1,
                                                n_features = self.hilbert.size,
                                                hidden_activation= nn.gelu,
                                                n_out = self.hilbert.n_orbitals * self.hilbert.n_fermions,
                                                    ),
                                    hilbert = self.hilbert,
                                    enforce_spin_flip=True)  

    def __call__(self, n: jax.Array) -> jax.Array:
        """Vectorized computation over batches."""
        return self.backflow(n)