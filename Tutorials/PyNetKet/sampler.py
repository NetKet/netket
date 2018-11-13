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

from __future__ import print_function
import netket as nk
import networkx as nx
import numpy as np
from mpi4py import MPI
import scipy.sparse as sparse

#Constructing a 1d lattice
g = nk.graph.Hypercube(L=4, ndim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

#Hamiltonian
ha = nk.operator.Ising(h=1.0, hilbert=hi)

#Machine
ma = nk.RbmSpin(hilbert=hi, alpha=1)
ma.InitRandomPars(seed=1234, sigma=0.1)

#Sampler
sa = nk.MetropolisLocal(machine=ma)
sa.Reset(True)
print(sa.Visible())
sa.Sweep()
print(sa.Visible())
