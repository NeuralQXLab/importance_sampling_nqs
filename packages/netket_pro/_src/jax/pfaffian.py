import jax.numpy as jnp
import jax


"""
The code in this file is taken from GPSKet
https://github.com/BoothGroup/GPSKet/blob/38582594c2f441b522aa4f4cf1a2e82b62c1660f/GPSKet/models/pfaffian.py#L102

I did not double check it, but they published a paper with it so hopefully it is correct.

We should double check it one day...
"""


# Legacy implementation which is slow
# """
# This implements a Pfaffian of a matrix as exemplified on WikiPedia, there are certainly better ways which we should adapt in the future,
# the derivation of this on WikiPedia does not explain why the trace identity can be carried over to non-positive matrices but the code seems to work.
# This approach is also not the numerically most stable one.
# See arxiv: 1102.3440 and the corresponding codebase (pfapack) for better implementations of the Pfaffian.
# TODO: Improve!
# """
# @jax.custom_jvp
# def log_pfaffian(mat):
#     n = mat.shape[0]//2
#     pauli_y = jnp.array([[0, -1.j], [1.j, 0.]])
#     vals = jnp.linalg.eigvals(jnp.dot(jnp.kron(pauli_y, jnp.eye(n)).T, mat))
#     return (0.5 * jnp.sum(jnp.log(vals)) + jnp.log(1.j) * (n**2))


"""
This implements the Pfaffian based on the Parlett-Reid algorithm as outlined in arxiv:1102.3440,
this implementation also borrows heavily from the corresponding codebase (pfapack, https://github.com/basnijholt/pfapack)
and is essentially just a reimplementation of its pfaffian_LTL method in jax.
The current implementation involves a for loop which will likely lead to sub-optimal compilation times when jitting this
but currently this seems to be the best solution to get around the jax limitations of requiring static loop counts.
"""


def log_pfaffian(mat):
    """
    Computes the logarithm of the Pfaffian of the input matrix. Has a custom JVP rule for the forward gradient.

    This function is based on the Parlett-Reid algorithm as outlined in
    `arxiv:1102.3440 <https://arxiv.org/pdf/1102.3440>`_ , following ideas in the
    `pfapack codebase <https://github.com/basnijholt/pfapack>`_ .

    The implementation relies on an internal for loop which is sup-optimal from the point of view of jax compilation time,
    but it's the only way known to humans nowdays to get it to work fast and support AD.

    We should try to use a more efficient implementation in the future, maybe based on pallas.

    .. note::

        This can be used to compute the Pair-Product geminal wavefunction which is a generalisation of a slater determinant,
        assuming we pass it a matrix of orbital amplitudes (much like with the Slater determinant).

    Args:
        mat: The input matrix for which to compute the Pfaffian.

    """
    return _log_pfaffian(mat)


@jax.custom_jvp
def _log_pfaffian(mat):
    # TODO: add some sanity checks here
    n = mat.shape[0] // 2
    matrix = mat.astype(jnp.complex128)
    value = 0.0
    for count in range(n):
        index = count * 2
        # permute rows/cols for numerical stability
        largest_index = jnp.argmax(jnp.abs(matrix[index + 1 :, index]))
        # exchange rows and columns
        updated_mat = matrix.at[index + 1, index:].set(
            matrix[index + largest_index + 1, index:]
        )
        updated_mat = updated_mat.at[index + largest_index + 1, index:].set(
            matrix[index + 1, index:]
        )
        matrix = updated_mat
        updated_mat = matrix.at[index:, index + 1].set(
            matrix[index:, index + largest_index + 1]
        )
        updated_mat = updated_mat.at[index:, index + largest_index + 1].set(
            matrix[index:, index + 1]
        )
        matrix = updated_mat
        # sign picked up
        value += jnp.where(largest_index != 0, jnp.log(-1 + 0.0j), 0.0)
        # value update
        value = jnp.where(
            matrix[index + 1, index] != 0.0,
            value + jnp.log(matrix[index, index + 1]),
            -jax.numpy.inf + 0.0j,
        )
        t = matrix[index, (index + 2) :] / matrix[index, index + 1]
        matrix = matrix.at[index + 2 :, index + 2 :].add(
            jnp.outer(t, matrix[index + 2 :, index + 1])
        )
        matrix = matrix.at[index + 2 :, index + 2 :].add(
            -jnp.outer(matrix[index + 2 :, index + 1], t)
        )
    return value


@_log_pfaffian.defjvp
def _log_pfaffian_jvp(primals, tangents):
    derivative = 0.5 * jnp.linalg.inv(primals[0]).T
    return (_log_pfaffian(primals[0]), derivative.flatten().dot(tangents[0].flatten()))
