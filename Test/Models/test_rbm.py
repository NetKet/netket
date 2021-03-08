# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import netket as nk
import jax.numpy as jnp

import pytest


@pytest.mark.parametrize("use_hidden_bias", [True, False])
@pytest.mark.parametrize("use_visible_bias", [True, False])
@pytest.mark.parametrize("kind", ["scan", "expand"])
def test_RBMSymm(use_hidden_bias, use_visible_bias, kind):
    N = 8
    hi = nk.hilbert.Spin(1 / 2, N)

    # Only translations, N_symm = N_sites
    perms1 = jnp.array([jnp.roll(jnp.arange(N), i) for i in range(N)])

    # All chain automorphisms, N_symm = 2 N_sites
    g = nk.graph.Chain(N)
    perms2 = jnp.array(g.automorphisms())

    for perms in perms1, perms2:
        ma = nk.models.RBMSymm(
            perms,
            alpha=4,
            use_visible_bias=use_visible_bias,
            use_bias=use_hidden_bias,
            bias_init=nk.nn.initializers.uniform(),
            visible_bias_init=nk.nn.initializers.uniform(),
        )
        pars = ma.init(nk.jax.PRNGKey(), hi.random_state(1))

        print(pars)

        v = hi.random_state(3)
        vals = [ma.apply(pars, v[..., p]) for p in perms]

        for val in vals:
            assert jnp.allclose(val, vals[0])
