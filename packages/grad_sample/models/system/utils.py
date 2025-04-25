# 2D spin models
import netket as nk
import numpy as np
from netket.utils.group._point_group import PGSymmetry, PointGroup
from netket.utils.group._semigroup import Identity
from netket.utils.types import Array


# Functions for defining symmetries of Shastry-Sutherland model
def reflect_and_translate(angle: float, t: Array) -> PGSymmetry:
    """
    Return `PGSymmetry` representing a translation followed by a 2D reflection across an axis at angle `angle` to the +x direction through the (0,0) point.
    This is equivalent to reflecting in the axis at angle `angle` passing through (0,0) and then translating by W@`t`, where W is
    the reflection matrix.
    Args:
        angle: the angle between the +x axis and the reflection axis.
        t: the translation vector
    """
    axis = np.radians(angle) * 2  # the mirror matrix is written in terms of 2φ
    W = np.asarray(
        [[np.cos(axis), np.sin(axis)], [np.sin(axis), -np.cos(axis)]]
    )  # the reflection matrix
    w = W @ t
    return PGSymmetry(W, w)  # transformation is W@x + w


def reflect_and_translate_group(angle: float, t: Array) -> PGSymmetry:
    """
    Returns the Z_2 `PointGroup`containing the identity and reflect_and_translate(angle,t)

    Arguments:
        trans: translation vector
        origin: a point on the glide axis, defaults to the origin
    The output is only a valid `PointGroup` after supplying a `unit_cell`
    consistent with the glide axis; otherwise, operations like `product_table`
    will fail.
    """
    return PointGroup([Identity(), reflect_and_translate(angle, t)], ndim=2)
