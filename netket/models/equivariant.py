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

from typing import Union, Optional, Tuple, Any, Callable, Iterable

import numpy as np

import jax
from jax import numpy as jnp
from flax import linen as nn

from netket.hilbert import AbstractHilbert
from netket.graph import AbstractGraph, SymmGroup
from netket.utils import HashableArray
from netket.utils.types import PRNGKeyT, Shape, DType, Array, NNInitFunc


from netket import nn as nknn
from netket.nn.initializers import lecun_complex, zeros, variance_scaling


class GCNN(nn.Module):
    """Implements a group convolutional neural network with symmetry
    averaging in the last layer as described in Roth et al. 2021."""

    symmetries: Union[HashableArray, SymmGroup]
    """A group of symmetry operations (or array of permutation indices) over which the layer should be invariant.
    Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`. 
    """
    layers: int
    """Number of layers (not including sum layer over output)"""
    features: Union[Tuple, int]
    """Number of features in each layer starting from the input. If
    a single number is given, all layers will have the same number
    of features"""
    group_algebra: HashableArray = None
    """A flattened array representing the mapping between relative and absolute poses.
    Does not to be specified if symmetries is a SymmGroup
    """
    dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.relu
    """The nonlinear activation function between hidden layers."""
    output_activation: Any = nknn.logcosh
    """The nonlinear activation before the output."""
    use_bias: bool = True
    """if True uses a bias in all layers."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = variance_scaling(1.0, "fan_in", "normal")
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias."""

    def setup(self):

        self.n_symm = np.asarray(self.symmetries).shape[0]

        if not np.any(self.group_algebra) and not isinstance(
            self.symmetries, SymmGroup
        ):
            raise AttributeError(
                "Group algebra must be specified if symmetries are given as an array"
            )

        if not np.any(self.group_algebra):
            group_algebra = self.symmetries.group_algebra()
        else:
            group_algebra = self.group_algebra

        if not np.asarray(group_algebra).shape[0] == np.square(self.n_symm):
            raise ValueError("Group algebra must have shape [n_symm*n_symm]")

        if isinstance(self.features, int):
            feature_dim = [self.features for layer in range(self.layers)]
        else:
            if not len(self.features) == self.layers:
                raise ValueError(
                    """Length of vector specifying feature dimensions must be the same as the number of layers"""
                )
            else:
                feature_dim = tuple(self.features)

        self.dense_symm = nknn.DenseSymm(
            symmetries=self.symmetries,
            features=feature_dim[0],
            dtype=self.dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            precision=self.precision,
        )

        self.equivariant_layers = [
            nknn.DenseEquivariant(
                group_algebra=group_algebra,
                in_features=feature_dim[layer],
                out_features=feature_dim[layer + 1],
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for layer in range(self.layers - 1)
        ]

    @nn.compact
    def __call__(self, x_in):
        x = self.dense_symm(x_in)
        for layer in range(self.layers - 1):
            x = self.activation(x)
            x = self.equivariant_layers[layer](x)

        x = self.output_activation(x)
        x = jnp.sum(x, axis=-1)

        return x