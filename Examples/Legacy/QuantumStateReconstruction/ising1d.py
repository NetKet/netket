# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from netket import legacy as nk
from generate_data import generate


# Load the data
N = 10
hi, rotations, training_samples, training_bases, ha, psi = generate(
    N, n_basis=2 * N, n_shots=500
)

# Machine
ma = nk.machine.RbmSpinPhase(hilbert=hi, alpha=1)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Sampler
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=1000)

# Optimizer
op = nk.optimizer.AdaDelta(ma)

sr = nk.optimizer.SR(ma, diag_shift=0.01)

# Quantum State Reconstruction
qst = nk.Qsr(
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    n_samples_data=500,
    rotations=rotations,
    samples=training_samples,
    bases=training_bases,
    sr=sr,
)

obs = {"Energy": ha}

qst.run(n_iter=2000, out="output", obs=obs)
