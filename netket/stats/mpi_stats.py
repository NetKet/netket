import numpy as np
import jax.numpy as jnp
from ._sum_inplace import sum_inplace as mpi_sum

from netket.utils import (
    n_nodes as _n_nodes,
)


def subtract_mean(x, axis=None):
    """
    Subtracts the mean of the input array over all but the last dimension
    and over all MPI processes from each entry.

    Args:
        x: Input array
        axis: Axis or axes along which the means are computed. The default (None) is to
              compute the mean of the flattened array.

    Returns:
        The resulting array.

    """
    x_mean = mean(x, axis=axis)
    x -= x_mean

    return x


def mean(a, axis=None, keepdims: bool = False):
    """
    Compute the arithmetic mean along the specified axis and over MPI processes.

    Returns the average of the array elements. The average is taken over the flattened array by default,
    otherwise over the specified axis. float64 intermediate and return values are used for integer inputs.

    Args:
        a: The input array
        axis: Axis or axes along which the means are computed. The default (None) is to
              compute the mean of the flattened array.
        keepdims: If True the output array will have the same number of dimensions as the input,
              with the reduced axes having length 1. (default=False)

    Returns:
        The array with reduced dimensions defined by axis.

    """
    out = a.mean(axis=axis, keepdims=keepdims)

    return mpi_sum(out) / _n_nodes


def sum(a, axis=None, keepdims: bool = False):
    """
    Compute the sum along the specified axis and over MPI processes.

    Args:
        a: The input array
        axis: Axis or axes along which the mean is computed. The default (None) is to
              compute the mean of the flattened array.
        out: An optional pre-allocated array to fill with the result.
        keepdims: If True the output array will have the same number of dimensions as the input,
              with the reduced axes having length 1. (default=False)

    Returns:
        The array with reduced dimensions defined by axis. If out is not none, returns out.

    """
    # if it's a numpy-like array...
    if hasattr(a, "shape"):
        # use jax
        a_sum = a.sum(axis=axis, keepdims=keepdims)
    else:
        # assume it's a scalar
        a_sum = jnp.asarray(a)

    return mpi_sum(a_sum)


def var(a, axis=None, ddof: int = 0):
    """
    Compute the variance mean along the specified axis and over MPI processes.

    Args:
        a: The input array
        axis: Axis or axes along which the variance is computed. The default (None) is to
              compute the variance of the whole flattened array.
        out: An optional pre-allocated array to fill with the result.
        ddof: “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
              where N represents the number of elements. By default ddof is zero.

    Returns:
        The array with reduced dimensions defined by axis. If out is not none, returns out.

    """
    m = mean(a, axis=axis)

    if axis is None:
        ssq = jnp.abs(a - m) ** 2.0
    else:
        ssq = jnp.abs(a - jnp.expand_dims(m, axis)) ** 2.0

    out = sum(ssq, axis=axis)

    n_all = total_size(a, axis=axis)
    out /= n_all - ddof

    return out


def total_size(a, axis=None):
    """
    Compute the total number of elements stored in the input array among all MPI processes.

    This function essentially returns MPI_sum_among_processes(a.size).

    Args:
        a: The input array.
        axis: If specified, only considers the total size of that axis.

    Returns:
        a.size or a.shape[axis], reduced among all MPI processes.
    """
    if axis is None:
        l_size = a.size
    else:
        l_size = a.shape[axis]

    return mpi_sum(l_size)
