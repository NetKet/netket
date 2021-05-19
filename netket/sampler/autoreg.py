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

from flax import struct
from netket.sampler import Sampler, SamplerState
from netket.utils.types import PRNGKeyT, PyTree

import jax
from jax import numpy as jnp


def vmap_choice(key, a, p, replace=True):
    """
    p.shape: (batch, a.shape)
    Return shape: (batch, )
    """

    def scan_fun(key, p_i):
        new_key, key = jax.random.split(key)
        out_i = jax.random.choice(key, a, replace=replace, p=p_i)
        return new_key, out_i

    _, out = jax.lax.scan(scan_fun, key, p)
    return out


@struct.dataclass
class ARSamplerState(SamplerState):
    σ: jnp.ndarray
    """current batch of (maybe partially sampled) configurations."""
    cache: PyTree
    """auxiliary states, e.g., used to implement fast autoregressive sampling."""
    key: PRNGKeyT
    """state of the random number generator."""

    def __repr__(self):
        return f"{type(self).__name__}(rng state={self.key})"


@struct.dataclass
class ARSampler(Sampler):
    """Sampler for autoregressive neural networks."""

    def _init_state(sampler, model, params, key):
        σ = jnp.zeros((sampler.n_chains, sampler.hilbert.size), dtype=sampler.dtype)
        cache = model.apply(
            params,
            σ,
            method=model.init_cache,
        )
        return ARSamplerState(σ=σ, cache=cache, key=key)

    def _reset(sampler, model, params, state):
        return state

    def _sample_next(sampler, model, params, state):
        def scan_fun(carry, index):
            σ, cache, key = carry
            new_key, key = jax.random.split(key)

            p, cache = model.apply(
                params,
                σ,
                cache,
                method=model.conditionals,
            )
            local_states = jnp.asarray(
                sampler.hilbert.local_states, dtype=sampler.dtype
            )
            p = p[:, index, :]
            new_σ = vmap_choice(key, local_states, p)
            σ = σ.at[:, index].set(new_σ)

            return (σ, cache, new_key), None

        new_key, key_init, key_scan = jax.random.split(state.key, 3)

        # Init `σ` and `cache` before generating each sample,
        # even if `params` is not changed and `reset` is not called
        σ = sampler.hilbert.random_state(
            key_init, size=sampler.n_batches, dtype=sampler.dtype
        )
        cache = model.apply(
            params,
            σ,
            method=model.init_cache,
        )

        indices = jnp.arange(sampler.hilbert.size)
        (σ, cache, _), _ = jax.lax.scan(
            scan_fun,
            (σ, cache, key_scan),
            indices,
        )

        new_state = state.replace(σ=σ, cache=cache, key=new_key)
        return new_state, σ
