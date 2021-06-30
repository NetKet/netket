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

import jax
from jax import numpy as jnp

from functools import partial

from flax.linen.initializers import *

lecun_complex = partial(variance_scaling, 1.0, "fan_in", "normal", dtype=jnp.complex64)


def unit_normal_scaling(key, shape, dtype):
    return jax.random.normal(key, shape, dtype) / jnp.sqrt(
        jnp.prod(jnp.asarray(shape[1:]))
    )
