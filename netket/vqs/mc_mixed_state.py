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

from typing import Optional, Callable, Union, Tuple

import numpy as np

import jax
from jax import numpy as jnp

from flax import linen as nn
from flax import serialization

import netket
from netket import jax as nkjax
from netket.sampler import Sampler
from netket.stats import Stats, statistics
from netket.utils import warn_deprecation
from netket.utils.types import PyTree, SeedT
from netket.operator import (
    AbstractOperator,
    local_cost_function,
    local_value_op_op_cost,
)

from .base import VariationalMixedState
from .mc_state import MCState

AFunType = Callable[[nn.Module, PyTree, jnp.ndarray], jnp.ndarray]
ATrainFunType = Callable[
    [nn.Module, PyTree, jnp.ndarray, Union[bool, PyTree]], jnp.ndarray
]


def apply_diagonal(bare_afun, w, x, *args, **kwargs):
    x = jnp.hstack((x, x))
    return bare_afun(w, x, *args, **kwargs)


class MCMixedState(VariationalMixedState, MCState):
    """Variational State for a Mixed Variational Neural Quantum State.

    The state is sampled according to the provided sampler, and its diagonal is sampled
    according to another sampler.
    """

    def __init__(
        self,
        sampler: Sampler,
        model=None,
        *,
        sampler_diag: Optional[Sampler] = None,
        n_samples_diag: Optional[int] = None,
        n_samples_per_rank_diag: Optional[int] = None,
        n_discard_diag: Optional[int] = None,  # deprecated
        n_discard_per_chain_diag: Optional[int] = None,
        variables: Optional[PyTree] = None,
        seed: Optional[SeedT] = None,
        sampler_seed: Optional[SeedT] = None,
        **kwargs,
    ):
        """
        Constructs the MCMixedState.
        Arguments are the same as :class:`MCState`.

        Arguments:
            sampler: the sampler.
            model: (optional) the model. If not provided, you must provide `variables` or `init_fun`, and `apply_fun`.
            sampler_diag: the sampler for the diagonal of the density matrix. Defaults to be the same type as `sampler`.

            n_samples: the total number of samples across chains and processes when sampling (default=1000).
            n_samples_per_rank: the total number of samples across chains on one process when sampling. Cannot be
                specified together with n_samples (default=None).
            n_discard_per_chain: number of discarded samples at the beginning of each Monte Carlo chain (default=0 for exact sampler,
                and n_samples/10 for approximate sampler).

            n_samples_diag: the total number of samples across chains and processes when sampling the diagonal (default=1000).
            n_samples_per_rank_diag: the total number of samples across chains on one process when sampling the diagonal.
                Cannot be specified together with `n_samples_diag` (default=None).
            n_discard_per_chain_diag: number of discarded samples at the beginning of each Monte Carlo chain when sampling
                the diagonal (default=0 for exact sampler, and n_samples/10 for approximate sampler).

            variables: parameters and mutable states of the model.
                See `Flax's module variables documentation <https://flax.readthedocs.io/en/latest/flax.linen.html#module-flax.core.variables>`_
                (default=None).

            init_fun: Function of the signature `f(model, rng_key, dummy_input) -> variables` used to initialise the variables.
                Defaults to `model.init(rng_key, dummy_input)`.
                Only specify if your model has a non-standard init method.
            apply_fun: Function of the signature `f(model, variables, σ) -> log_psi` used to evaluate the model.
                Defaults to `model.apply(variables, σ)`.
                Only specify if your model has a non-standard apply method.

            seed: rng seed used to generate the parameters of the model (only if `variables` is not passed). Defaults to a random one.
            sampler_seed: rng seed used to initialise the sampler. Defaults to a random one.

            mutable: Specifies which variable collections of the model should
                be treated as mutable. bool: all/no collections are mutable. str: The name of a
                single mutable collection. list: A list of names of mutable collections.
                This is used to mutate the state of the model while you train it (for example
                to implement BatchNorm).
                See `Flax's Module.apply documentation <https://flax.readthedocs.io/en/latest/_modules/flax/linen/module.html#Module.apply>`_
                (default=False).
        """
        if seed is None:
            seed_diag = None
        else:
            seed, seed_diag = jax.random.split(nkjax.PRNGKey(seed))

        if sampler_seed is None:
            sampler_seed_diag = None
        else:
            sampler_seed, sampler_seed_diag = jax.random.split(
                nkjax.PRNGKey(sampler_seed)
            )

        self._diagonal = None

        hilbert_physical = sampler.hilbert.physical

        super().__init__(
            sampler.hilbert.physical,
            sampler,
            model,
            variables=variables,
            seed=seed,
            sampler_seed=sampler_seed,
            **kwargs,
        )

        if sampler_diag is None:
            sampler_diag = sampler.replace(hilbert=hilbert_physical)

        sampler_diag = sampler_diag.replace(machine_pow=1)

        diagonal_apply_fun = nkjax.HashablePartial(apply_diagonal, self._apply_fun)

        for kw in [
            "n_samples",
            "n_discard",
            "n_discard_per_chain",
        ]:  # TODO: remove n_discard after deprecation.
            if kw in kwargs:
                kwargs.pop(kw)

        # TODO: remove deprecation.
        if n_discard_diag is not None and n_discard_per_chain_diag is not None:
            raise ValueError(
                "`n_discard_diag` has been renamed to `n_discard_per_chain_diag` and deprecated."
                "Specify only `n_discard_per_chain_diag`."
            )
        elif n_discard_diag is not None:
            warn_deprecation(
                "`n_discard_diag` has been renamed to `n_discard_per_chain_diag` and deprecated."
                "Please update your code to `n_discard_per_chain_diag`."
            )
            n_discard_per_chain_diag = n_discard_diag

        self._diagonal = MCState(
            sampler_diag,
            apply_fun=diagonal_apply_fun,
            n_samples=n_samples_diag,
            n_samples_per_rank=n_samples_per_rank_diag,
            n_discard_per_chain=n_discard_per_chain_diag,
            variables=self.variables,
            seed=seed_diag,
            sampler_seed=sampler_seed_diag,
            **kwargs,
        )

    @property
    def diagonal(self):
        return self._diagonal

    @property
    def sampler_diag(self) -> Sampler:
        """The Monte Carlo sampler used by this Monte Carlo variational state to
        sample the diagonal."""
        return self.diagonal.sampler

    @sampler_diag.setter
    def sampler_diag(self, sampler):
        self.diagonal.sampler = sampler

    @property
    def n_samples_diag(self) -> int:
        """The total number of samples generated at every sampling step
        when sampling the diagonal of this mixed state.
        """
        return self.diagonal.n_samples

    @n_samples_diag.setter
    def n_samples_diag(self, n_samples):
        self.diagonal.n_samples = n_samples

    @property
    def chain_length_diag(self) -> int:
        """
        Length of the markov chain used for sampling the diagonal configurations.

        If running under MPI, the total samples will be n_nodes * chain_length * n_batches.
        """
        return self.diagonal.chain_length

    @chain_length_diag.setter
    def chain_length_diag(self, length: int):
        self.diagonal.chain_length = length

    @property
    def n_discard_per_chain_diag(self) -> int:
        """Number of discarded samples at the beginning of the markov chain used to
        sample the diagonal of this mixed state.
        """
        return self.diagonal.n_discard_per_chain

    @n_discard_per_chain_diag.setter
    def n_discard_per_chain_diag(self, n_discard_per_chain: Optional[int]):
        self.diagonal.n_discard_per_chain = n_discard_per_chain

    # TODO: deprecate
    @property
    def n_discard_diag(self) -> int:
        """
        DEPRECATED: Use `n_discard_per_chain_diag` instead.

        Number of discarded samples at the beginning of the markov chain.
        """
        warn_deprecation(
            "`n_discard_diag` has been renamed to `n_discard_per_chain_diag` and deprecated."
            "Please update your code to use `n_discard_per_chain_diag`."
        )

        return self.n_discard_per_chain_diag

    @n_discard_diag.setter
    def n_discard_diag(self, val) -> int:
        warn_deprecation(
            "`n_discard_diag` has been renamed to `n_discard_per_chain_diag` and deprecated."
            "Please update your code to use `n_discard_per_chain_diag`."
        )
        self.n_discard_per_chain_diag = val

    @MCState.parameters.setter
    def parameters(self, pars: PyTree):
        MCState.parameters.fset(self, pars)
        if self.diagonal is not None:
            self.diagonal.parameters = pars

    @MCState.model_state.setter
    def model_state(self, state: PyTree):
        MCState.model_state.fset(self, state)
        if self.diagonal is not None:
            self.diagonal.model_state = state

    def reset(self):
        super().reset()
        if self.diagonal is not None:
            self.diagonal.reset()

    def expect_operator(self, Ô: AbstractOperator) -> Stats:
        σ = self.diagonal.samples
        σ_shape = σ.shape
        σ = σ.reshape((-1, σ.shape[-1]))

        σ_np = np.asarray(σ)
        σp, mels = Ô.get_conn_padded(σ_np)

        # now we have to concatenate the two
        O_loc = local_cost_function(
            local_value_op_op_cost,
            self._apply_fun,
            self.variables,
            σp,
            mels,
            σ,
        ).reshape(σ_shape[:-1])

        # notice that loc.T is passed to statistics, since that function assumes
        # that the first index is the batch index.
        return statistics(O_loc.T)

    def expect_and_grad_operator(
        self, Ô: AbstractOperator, is_hermitian=None
    ) -> Tuple[Stats, PyTree]:
        raise NotImplementedError

    def to_matrix(self, normalize: bool = True) -> jnp.ndarray:
        return netket.nn.to_matrix(
            self.hilbert, self._apply_fun, self.variables, normalize=normalize
        )

    def __repr__(self):
        return (
            "MCMixedState("
            + "\n  hilbert = {},".format(self.hilbert)
            + "\n  sampler = {},".format(self.sampler)
            + "\n  n_samples = {},".format(self.n_samples)
            + "\n  n_discard_per_chain = {},".format(self.n_discard_per_chain)
            + "\n  sampler_state = {},".format(self.sampler_state)
            + "\n  sampler_diag = {},".format(self.sampler_diag)
            + "\n  n_samples_diag = {},".format(self.n_samples_diag)
            + "\n  n_discard_per_chain_diag = {},".format(self.n_discard_per_chain_diag)
            + "\n  sampler_state_diag = {},".format(self.diagonal.sampler_state)
            + "\n  n_parameters = {})".format(self.n_parameters)
        )

    def __str__(self):
        return (
            "MCMixedState("
            + "hilbert = {}, ".format(self.hilbert)
            + "sampler = {}, ".format(self.sampler)
            + "n_samples = {})".format(self.n_samples)
        )


# serialization


def serialize_MCMixedState(vstate):
    state_dict = {
        "variables": serialization.to_state_dict(vstate.variables),
        "sampler_state": serialization.to_state_dict(vstate.sampler_state),
        "diagonal": serialization.to_state_dict(vstate.diagonal),
        "n_samples": vstate.n_samples,
        "n_discard_per_chain": vstate.n_discard_per_chain,
    }
    return state_dict


def deserialize_MCMixedState(vstate, state_dict):
    import copy

    new_vstate = copy.copy(vstate)
    new_vstate.reset()

    # restore the diagonal first so we can relink the samples
    new_vstate._diagonal = serialization.from_state_dict(
        vstate._diagonal, state_dict["diagonal"]
    )

    new_vstate.variables = serialization.from_state_dict(
        vstate.variables, state_dict["variables"]
    )
    new_vstate.sampler_state = serialization.from_state_dict(
        vstate.sampler_state, state_dict["sampler_state"]
    )
    new_vstate.n_samples = state_dict["n_samples"]
    new_vstate.n_discard_per_chain = state_dict["n_discard_per_chain"]

    return new_vstate


serialization.register_serialization_state(
    MCMixedState,
    serialize_MCMixedState,
    deserialize_MCMixedState,
)
