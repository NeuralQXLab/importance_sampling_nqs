import jax
import numpy as np


def variables_are_equal(var1, var2):
    """
    Compare the variables of a network, returned by net.init(...).
    Returns True if equivalent values and structure of pytree
    """
    vals = []
    treedefs = []
    for var in (var1, var2):
        val, treedef = jax.tree.flatten(var)
        vals.append(val)
        treedefs.append(treedef)

    flag = treedefs[0] == treedefs[1]

    for arr1, arr2 in zip(vals[0], vals[1]):
        is_equal = np.all(arr1 == arr2)
        flag *= is_equal

    return flag
