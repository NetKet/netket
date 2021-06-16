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

from typing import Any, Callable, Iterable, Union

import jax
from jax import numpy as jnp
from netket.hilbert import CustomHilbert
from netket.models.autoreg import ARNN, l2_normalize
from netket.nn import FastMaskedConv1D, FastMaskedDense1D
from netket.nn.initializers import zeros
from netket.nn.masked_linear import default_kernel_init
from netket.utils.types import Array, DType, NNInitFunc


class FastARNNDense(ARNN):
    """
    Fast autoregressive neural network with dense layers.
    See :ref:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    TODO: FastMaskedDense1D does not support JIT yet
    """

    hilbert: CustomHilbert
    """the discrete Hilbert space."""
    layers: int
    """number of layers."""
    features: Union[Iterable[int], int]
    """number of features in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    activation: Callable[[Array], Array] = jax.nn.selu
    """the nonlinear activation function between hidden layers (default: selu)."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    dtype: DType = jnp.float64
    """the dtype of the weights (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    eps: float = 1e-7
    """a small number to avoid numerical instability."""

    def setup(self):
        if isinstance(self.features, int):
            features = [self.features] * (self.layers - 1) + [self.hilbert.local_size]
        else:
            features = self.features
        assert len(features) == self.layers
        assert features[-1] == self.hilbert.local_size

        self._layers = [
            FastMaskedDense1D(
                size=self.hilbert.size,
                features=features[i],
                exclusive=(i == 0),
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]

    def conditional(self, inputs: Array, index: int) -> Array:
        return _conditional(self, inputs, index)

    def conditionals(self, inputs: Array) -> Array:
        return _conditionals(self, inputs)

    def __call__(self, inputs: Array) -> Array:
        return _call(self, inputs)


class FastARNNConv1D(ARNN):
    """
    Fast autoregressive neural network with 1D convolution layers.
    See :ref:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    """

    hilbert: CustomHilbert
    """the discrete Hilbert space."""
    layers: int
    """number of layers."""
    features: Union[Iterable[int], int]
    """number of features in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    kernel_size: int
    """length of the convolutional kernel."""
    kernel_dilation: int = 1
    """dilation factor of the convolution kernel (default: 1)."""
    activation: Callable[[Array], Array] = jax.nn.selu
    """the nonlinear activation function between hidden layers (default: selu)."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    dtype: DType = jnp.float64
    """the dtype of the weights (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    eps: float = 1e-7
    """a small number to avoid numerical instability."""

    def setup(self):
        if isinstance(self.features, int):
            features = [self.features] * (self.layers - 1) + [self.hilbert.local_size]
        else:
            features = self.features
        assert len(features) == self.layers
        assert features[-1] == self.hilbert.local_size

        self._layers = [
            FastMaskedConv1D(
                size=self.hilbert.size,
                features=features[i],
                kernel_size=self.kernel_size,
                kernel_dilation=self.kernel_dilation,
                exclusive=(i == 0),
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]

    def conditional(self, inputs: Array, index: int) -> Array:
        return _conditional(self, inputs, index)

    def conditionals(self, inputs: Array) -> Array:
        return _conditionals(self, inputs)

    def __call__(self, inputs: Array) -> Array:
        return _call(self, inputs)


def _conditional(model: ARNN, inputs: Array, index: int) -> Array:
    """
    Computes the conditional probabilities for a site to take each value.
    See `ARNN.conditional`.
    """
    if inputs.ndim == 1:
        inputs = jnp.expand_dims(inputs, axis=0)

    # When `index = 0`, it doesn't matter what slice of `x` we take
    x = inputs[:, index - 1, None]

    for i in range(model.layers):
        if i > 0:
            x = model.activation(x)
        x = model._layers[i](x, index)

    log_psi = l2_normalize(x, model.eps)
    p = jnp.exp(2 * log_psi.real)
    return p


def _conditionals_log_psi(model: ARNN, inputs: Array) -> Array:
    """
    Computes the log of the conditional wave-functions for each site if it takes each value.
    See `ARNN.conditionals`.
    """
    x = jnp.expand_dims(inputs, axis=-1)

    for i in range(model.layers):
        if i > 0:
            x = model.activation(x)
        x = model._layers[i].eval_full(x)

    log_psi = l2_normalize(x, model.eps)
    return log_psi


def _conditionals(model: ARNN, inputs: Array) -> Array:
    """
    Computes the conditional probabilities for each site to take each value.
    See `ARNN.conditionals`.
    """
    if inputs.ndim == 1:
        inputs = jnp.expand_dims(inputs, axis=0)

    log_psi = _conditionals_log_psi(model, inputs)

    p = jnp.exp(2 * log_psi.real)
    return p


def _call(model: ARNN, inputs: Array) -> Array:
    """Returns log_psi."""

    if inputs.ndim == 1:
        inputs = jnp.expand_dims(inputs, axis=0)

    initializing = model.is_mutable_collection("params")
    if initializing:
        # Create cache
        x = inputs[:, 0, None]
        for layer in model._layers:
            x = layer(x, 0)

    idx = (inputs + model.hilbert.local_size - 1) / 2
    idx = jnp.asarray(idx, jnp.int64)
    idx = jnp.expand_dims(idx, axis=-1)

    log_psi = _conditionals_log_psi(model, inputs)

    log_psi = jnp.take_along_axis(log_psi, idx, axis=-1)
    log_psi = log_psi.reshape((inputs.shape[0], -1)).sum(axis=1)
    return log_psi
