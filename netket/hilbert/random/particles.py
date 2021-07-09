import jax
from jax import numpy as jnp

from netket.hilbert import Particles
from netket.utils.dispatch import dispatch


@dispatch
def random_state(hilb: Particles, key, batches: int, *, dtype, width = 0.2):
    """Positions particles w.r.t. normal distribution,
     if no periodic boundary conditions are applied
    in a spatial dimension. Otherwise the particles are
     positioned evenly along the box from 0 to L, with Gaussian noise
     of certain width."""
    gaussian = jax.random.normal(key, shape=(batches, hilb.size))*width

    minL = jnp.min(jnp.where(jnp.array(hilb.L) is None, jnp.inf, hilb.L))
    uniform = jnp.tile(jnp.linspace(0., minL, hilb.size), (batches, 1))

    boundary = jnp.tile(jnp.array(hilb.N * hilb.L))

    rs = jnp.where(boundary is None, gaussian, uniform + gaussian)

    return jnp.asarray(rs, dtype=dtype)
