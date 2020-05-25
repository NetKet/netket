import numpy as _np
from numba import jit, objmode
from netket.utils import (
    n_nodes as _n_nodes,
    node_number as _rank,
    MPI_comm as _MPI_comm,
)


@jit
def seed(seed=None):
    """ Seed the random number generator. Each MPI process is automatically assigned
        a different, process-dependent, sub-seed.

        Parameters:
                  seed (int, optional): Seed for the randon number generator.

    """
    with objmode(derived_seed="int64"):
        size = _n_nodes
        rank = _rank

        if rank == 0:
            _np.random.seed(seed)
            derived_seed = _np.random.randint(0, 1 << 32, size=size)
        else:
            derived_seed = None

        if _n_nodes > 1:
            derived_seed = _MPI_comm.scatter(derived_seed, root=0)

    _np.random.seed(derived_seed)


@jit
def uniform(low=0.0, high=1.0):
    """
    Draw samples from a uniform distribution. Samples are uniformly distributed
    over the half-open interval [low, high) (includes low, but excludes high).

    Parameters:
              low (float, optional): Lower boundary of the output interval.
                                     All values generated will be greater than
                                     or equal to low. The default value is 0.
              high (float, optional): Upper boundary of the output interval.
                                     All values generated will be less than high.
                                     The default value is 1.0.
    Returns:
              float: A randon number uniformly distributed in [low,high).

    """
    return _np.random.uniform(low, high)


@jit
def randint(low, high):
    """
    Generate random integers from low (inclusive) to high (exclusive).

    Args:
        low (int): Lowest (signed) integer to be drawn from the distribution.
        high (int): One above the largest (signed) integer to be drawn from the distribution.

    Returns:
        int: A random integer uniformely distributed in [low,high).

    """
    return _np.random.randint(low, high)


def choice(a, size=None, replace=True, p=None):
    # TODO use always numpy version when argument p is made available in numba
    """
    Generates a random sample from a given 1-D array.

    Args:
        a (1-D array-like): A random sample is generated from its elements.
        replace (boolean, optional): Whether the sample is with or without replacement.
        p (1-D array-like, optional): The probabilities associated with each entry in a.
                    If not given the sample assumes a uniform distribution over all entries in a.

    Returns:
        single item or ndarry: The generated random samples
    """
    return _np.random.choice(a, size, replace, p)


# By default, the generator is initialized with a random seed (on node 0)
# and then propagated correctly to the other nodes
seed(None)
