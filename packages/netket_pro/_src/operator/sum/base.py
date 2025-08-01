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

from typing import Union
from collections.abc import Iterable

from abc import ABC

import jax.numpy as jnp

from netket.hilbert import AbstractHilbert
from netket.utils.types import Array
from netket.jax import canonicalize_dtypes
from netket.utils.numbers import is_scalar

from netket.operator import AbstractOperator, DiscreteJaxOperator, ContinuousOperator


def _flatten_sumoperators(operators: Iterable[AbstractOperator], coefficients: Array):
    """Flatten sumoperators inside of operators."""
    new_operators = []
    new_coeffs = []
    for op, c in zip(operators, coefficients):
        if isinstance(op, SumOperator):
            new_operators.extend(op.operators)
            new_coeffs.extend(c * op.coefficients)
        else:
            new_operators.append(op)
            new_coeffs.append(c)
    return new_operators, new_coeffs


class SumOperator(ABC):
    def __new__(cls, *args, **kwargs):
        # This logic overrides the constructor, such that if someone tries to
        # construct this class directly by calling `SumOperator(...)`
        # it will construct either a DiscreteHilbert or TensorDiscreteHilbert
        from .operator import SumGenericOperator
        from .discrete_jax_operator import SumDiscreteJaxOperator
        from .continuous import SumContinuousOperator

        if cls is SumOperator:
            if all(isinstance(op, DiscreteJaxOperator) for op in args):
                cls = SumDiscreteJaxOperator
            # elif all(isinstance(op, DiscreteOperator) for op in args):
            #    cls = SumDiscreteOperator
            elif all(isinstance(op, ContinuousOperator) for op in args):
                cls = SumContinuousOperator
            else:
                cls = SumGenericOperator
        return super().__new__(cls)

    def __init__(
        self,
        operators: Iterable[AbstractHilbert],
        *args,
        coefficients: Union[float, Iterable[float]] = 1.0,
        dtype=None,
        **kwargs,
    ):
        r"""Constructs a Sum of Operators.

        Args:
            *hilb: An iterable object containing at least 1 hilbert space.
        """
        hi_spaces = [op.hilbert for op in operators]
        if not all(hi == hi_spaces[0] for hi in hi_spaces):
            raise NotImplementedError(
                "Cannot construct a SumOperator for operators on different Hilbert Spaces"
            )

        if is_scalar(coefficients):
            coefficients = [coefficients for _ in operators]

        if len(operators) != len(coefficients):
            raise AssertionError("Each operator needs a coefficient")

        operators, coefficients = _flatten_sumoperators(operators, coefficients)

        dtype = canonicalize_dtypes(float, *operators, *coefficients, dtype=dtype)

        self._operators = tuple(operators)
        self._coefficients = jnp.asarray(coefficients, dtype=dtype)
        self._dtype = dtype

        super().__init__(
            *args, **kwargs
        )  # forwards all unused arguments so that this class is a mixin.

    @property
    def dtype(self):
        return self._dtype

    @property
    def operators(self) -> tuple[AbstractOperator, ...]:
        """The tuple of all operators in the terms of this sum. Every
        operator is summed with a corresponding coefficient
        """
        return self._operators

    @property
    def coefficients(self) -> tuple[AbstractOperator, ...]:
        return self._coefficients

    def __repr__(self) -> str:
        strs = [f"{type(self).__name__} with terms:"]
        for op, c in zip(self.operators, self.coefficients):
            strs.append(f" ∙ {c} * {op}")
        return "\n".join(strs)

    def __add__(self, other):
        if isinstance(other, SumOperator):
            ops = self.operators + other.operators
            coeffs = jnp.concatenate([self.coefficients, other.coefficents])
            dtype = self.dtype if self.dtype == other.dtype else None
        else:
            ops = (*self.operators, other)
            coeffs = jnp.concatenate([self.coefficients, jnp.array([1.0])])
            dtype = self.dtype if self.dtype == other.dtype else None

        return SumOperator(*ops, coefficients=coeffs, dtype=dtype)
