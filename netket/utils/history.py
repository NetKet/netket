import numpy as np
from numbers import Number


class History:
    """
    A class to store a time-series of scalar data.

    It has two member variables, `iter` and `values`.
    The first stores the `time` of the time series, while `values`
    stores the values at each iteration.
    """

    def __init__(self, values=[], iters=None, dtype=None, iter_dtype=None):
        if isinstance(values, Number):
            values = np.array([values], dtype=dtype)

        if iters is None:
            if iter_dtype is None:
                iter_dtype = np.int32
            iters = np.arange(len(values), dtype=iter_dtype)

        elif isinstance(iters, Number):
            iters = np.array([iters], dtype=iter_dtype)

        if len(values) != len(iters):
            raise ErrorException("Not matching lengths")

        self.iters = np.array(iters, dtype=iter_dtype)
        self.values = np.array(values, dtype=dtype)

    def append(self, val, it=None):
        """
        Append another value to this history object.

        Args:
            val: the value in the next timestep
            it: the time corresponding to this new value. If
                not defined, increment by 1.
        """
        if isinstance(val, History):
            self.values = np.concatenate([self.values, val.values])
            self.iters = np.concatenate([self.iters, val.iters])
            return

        try:
            self.values.resize(len(self.values) + 1)
        except:
            self.values = np.resize(self.values, (len(self.values) + 1))

        try:
            self.iters.resize(len(self.iters) + 1)
        except:
            self.iters = np.resize(self.iters, (len(self.iters) + 1))

        if it is None:
            it = self.iters[-1] - self.iters[-2]

        self.values[-1] = val
        self.iters[-1] = it

    def get(self):
        """
        Returns a tuple containing times and values of this history object
        """
        return self.iters, self.values

    def to_dict(self):
        """
        Converts the history object to dict.

        Used for serialization
        """
        return {"iters": self.iters, "values": self.values}

    def __array__(self, *args, **kwargs):
        """
        Automatically transform this object to a numpy array when calling
        asarray, by only considering the values and neglecting the times.
        """
        return np.array(self.values, *args, **kwargs)

    def __iter__(self):
        """
        You can iterate the values in history object.
        """
        """ Returns the Iterator object """
        return iter(zip(self.iters, self.values))


from functools import partial
from jax.tree_util import tree_map


def accum_in_tree(fun, tree_accum, tree, **kwargs):
    """
    Maps all the leafs in the two trees, applying the function with the leafs of tree1
    as first argument and the leafs of tree2 as second argument
    Any additional argument after the first two is forwarded to the function call.

    This is usefull e.g. to sum the leafs of two trees

    Args:
        fun: the function to apply to all leafs
        tree1: the structure containing leafs. This can also be just a leaf
        tree2: the structure containing leafs. This can also be just a leaf
        *args: additional positional arguments passed to fun
        **kwargs: additional kw arguments passed to fun

    Returns:
        An equivalent tree, containing the result of the function call.
    """
    if tree is None:
        return tree_accum

    elif isinstance(tree, list):
        if tree_accum is None:
            tree_accum = [None for _ in range(len(tree))]

        return [
            accum_in_tree(fun, _accum, _tree, **kwargs)
            for _accum, _tree in zip(tree_accum, tree)
        ]
    elif isinstance(tree, tuple):
        if tree_accum is None:
            tree_accum = (None for _ in range(len(tree)))

        return tuple(
            accum_in_tree(fun, _accum, _tree, **kwargs)
            for _accum, _tree in zip(tree_accum, tree)
        )
    elif isinstance(tree, dict):
        if tree_accum is None:
            tree_accum = {}

        for key in tree.keys():
            tree_accum[key] = accum_in_tree(
                fun, tree_accum.get(key, None), tree[key], **kwargs
            )

        return tree_accum
    elif hasattr(tree, "to_dict"):
        return accum_in_tree(fun, tree_accum, tree.to_dict(), **kwargs)
    else:
        return fun(tree_accum, tree, **kwargs)


def accum_histories(accum, data, *, step=0):
    if accum is None:
        return History([data], step)
    else:
        accum.append(data, it=step)
        return accum


accum_histories_in_tree = partial(accum_in_tree, accum_histories)
