from typing import Callable
from tqdm import tqdm
from copy import copy

import numpy as np
import scipy.sparse as sp

from netket.operator import LocalOperator
from netket.utils.types import Array
from typing import Union

from netket.vqs import VariationalState
from netket.operator import AbstractOperator
from netket_pro._src.util.diagonal_splitting import (
    split_hamiltonian as _split_hamiltonian_netket,
)

Operator = np.ndarray | AbstractOperator
ApplyFunT = Callable[[np.ndarray], np.ndarray]


def integrator(
    psi0: np.ndarray | VariationalState,
    e_ops: list[Operator, ...],
    tf: float,
    dt: float,
    f_apply: ApplyFunT,
    save: bool = True,
):
    """
    Solves the Schroedinger's equation using the following parameters:

    Args:
        psi0: initial state
        e_ops: a list of operators for expectation values to compute
        tf: final time
        dt: timestep
        f_apply: the update to perform at every time-step. Obtained from the epe solvers
        save: whether to save the wavefunction and observables along the calculation.
    """

    if isinstance(psi0, VariationalState):
        psi0 = psi0.to_array()
    if not isinstance(e_ops, list | tuple):
        raise TypeError("e_ops must be a list or a tuple of operators.")

    # Convert from netket format if they are netket operators
    e_ops = [op.to_sparse() if isinstance(op, AbstractOperator) else op for op in e_ops]

    psi = copy(psi0)

    psi_list = []
    e_ops_list = []

    _step = f_apply(dt)

    t_l = np.arange(0, tf, dt)
    for t in tqdm(t_l):
        if save:
            psi_list.append(psi)
            e_ops_list.append(
                [
                    np.vdot(
                        psi / np.linalg.norm(psi), op.dot(psi / np.linalg.norm(psi))
                    )
                    for op in e_ops
                ]
            )

        psi = _step(psi)
        psi = psi / np.linalg.norm(psi)

    psi_list.append(psi)
    e_ops_list.append([np.vdot(psi, op.dot(psi)) for op in e_ops])

    return psi_list, np.array(e_ops_list)


def _split_hamiltonian(H: Union[Array, LocalOperator], *, to_sparse: bool = True):
    if isinstance(H, Union[Array, sp.spmatrix]):
        return _split_hamiltonian_array(H)
    elif isinstance(H, LocalOperator):
        Hd, Ho = _split_hamiltonian_netket(H)
        if to_sparse:
            return Hd.to_sparse(), Ho.to_sparse()
        else:
            return Hd, Ho
    else:
        raise ValueError("Unknown type: ", type(H))


def _split_hamiltonian_array(H: Union[Array, sp.spmatrix]):
    H_diag = H.diagonal()
    H_offdiag = H - sp.diags(H_diag)
    return H_diag, H_offdiag
