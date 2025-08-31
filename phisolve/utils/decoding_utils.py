import numpy as np
import jax.numpy as jnp
from jax import vmap


def spin_to_bitstring(spin_list):
    spin_list = jnp.array(spin_list)
    spin_to_binary = lambda spin: jnp.where(spin == 1, 0.0, 1.0)
    return vmap(spin_to_binary)(spin_list)

def affine_trans(y, l, u):
    """
    Translate y in [-1, 1] to [l, u]
    Here 1 -> l,   -1 -> u.
    """
    return (l - u) / 2 * y + (l + u) / 2

def spin_to_box(spin_list, lb, ub):
    spin_list = jnp.array(spin_list)
    return vmap(affine_trans)(spin_list, lb, ub)