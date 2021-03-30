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

import netket as nk
import jax.numpy as jnp

from test_nn import _setup_symm

import pytest


@pytest.mark.parametrize("use_hidden_bias", [True, False])
@pytest.mark.parametrize("use_visible_bias", [True, False])
@pytest.mark.parametrize("permutations", ["trans", "autom"])
def test_RBMSymm(use_hidden_bias, use_visible_bias, permutations):
    g, hi, perms = _setup_symm(permutations, N=8)

    ma = nk.models.RBMSymm(
        permutations=perms,
        alpha=4,
        use_visible_bias=use_visible_bias,
        use_hidden_bias=use_hidden_bias,
        hidden_bias_init=nk.nn.initializers.uniform(),
        visible_bias_init=nk.nn.initializers.uniform(),
    )
    pars = ma.init(nk.jax.PRNGKey(), hi.random_state(1))

    print(pars)

    v = hi.random_state(3)
    vals = [ma.apply(pars, v[..., p]) for p in perms]

    for val in vals:
        assert jnp.allclose(val, vals[0])

    vmc = nk.VMC(
        nk.operator.Ising(hi, g, h=1.0),
        nk.optim.Sgd(0.1),
        nk.sampler.MetropolisLocal(hi),
        ma,
    )
    vmc.advance(1)


def test_RBMSymm_creation():
    hi = nk.hilbert.Spin(1 / 2, N=8)

    def check_init(creator):
        ma = creator()
        p = ma.init(nk.jax.PRNGKey(0), hi.numbers_to_states(0))

    perms = [[0, 1, 2, 3, 4, 5, 6, 7]]

    # Test different permutation argument types
    check_init(lambda: nk.models.RBMSymm(permutations=perms))
    check_init(lambda: nk.models.RBMSymm(permutations=jnp.array(perms)))
    check_init(lambda: nk.models.RBMSymm(permutations=lambda: jnp.array(perms)))

    # wrong shape
    with pytest.raises(ValueError):
        check_init(lambda: nk.models.RBMSymm(permutations=perms[0]))

    # init with graph
    check_init(lambda: nk.models.RBMSymm(permutations=nk.graph.Chain(8), alpha=2))

    # alpha too small
    with pytest.raises(ValueError):
        check_init(
            lambda: nk.models.RBMSymm(permutations=nk.graph.Hypercube(8, 2), alpha=1)
        )


@pytest.mark.parametrize("use_hidden_bias", [True, False])
@pytest.mark.parametrize("use_visible_bias", [True, False])
def test_RBMMultiVal(use_hidden_bias, use_visible_bias):
    N = 8
    M = 3
    hi = nk.hilbert.Fock(M, N)
    g = nk.graph.Chain(N)

    ma = nk.models.RBMMultiVal(
        alpha=2,
        n_classes=M + 1,
        use_visible_bias=use_visible_bias,
        use_hidden_bias=use_hidden_bias,
        hidden_bias_init=nk.nn.initializers.uniform(),
        visible_bias_init=nk.nn.initializers.uniform(),
    )
    pars = ma.init(nk.jax.PRNGKey(), hi.random_state(1))

    vmc = nk.VMC(
        nk.operator.BoseHubbard(hi, g, U=1.0),
        nk.optim.Sgd(0.1),
        nk.sampler.MetropolisLocal(hi),
        ma,
    )
    vmc.advance(1)
