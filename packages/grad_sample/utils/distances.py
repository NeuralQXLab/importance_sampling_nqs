import jax.numpy as jnp

def fs_dist(dp_exact, dp_approx):
    # Fubini Study distance on hilbert space
    dot = jnp.dot(dp_approx.conj(),dp_exact)*jnp.dot(dp_approx, dp_exact.conj())
    norm = (jnp.dot(dp_approx, dp_approx.conj())*jnp.dot(dp_exact.conj(), dp_exact))
    return (jnp.arccos(jnp.sqrt((dot/norm).real))**2)

def dot_prod(dp_exact, dp_approx):
    # Fubini Study distance on hilbert space
    dot = jnp.dot(dp_approx.conj(),dp_exact)*jnp.dot(dp_approx, dp_exact.conj())
    norm = (jnp.dot(dp_approx, dp_approx.conj())*jnp.dot(dp_exact.conj(), dp_exact))
    return (dot/norm).real

def curved_dist(dp_exact, dp_approx, S):
    # distance induced by the S matrix
    dot = jnp.dot(dp_approx.conj(), S @ dp_exact)*jnp.dot(dp_exact.conj(), S @ dp_approx )
    norm = (jnp.dot(dp_approx.conj(), S @ dp_approx)*jnp.dot(dp_exact.conj(), S @ dp_exact))
    return (jnp.arccos(jnp.sqrt((dot/norm).real))**2) # distance on parameter space with S matrix


def param_overlap(dp_exact, dp_approx, S):
    # distance induced by the S matrix
    dot = jnp.dot(dp_approx.conj(), S @ dp_exact)*jnp.dot(dp_exact.conj(), S @ dp_approx )
    norm = (jnp.dot(dp_approx.conj(), S @ dp_approx)*jnp.dot(dp_exact.conj(), S @ dp_exact))
    return jnp.sqrt((dot/norm).real)# distance on parameter space with S matrix
