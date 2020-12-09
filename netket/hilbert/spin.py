from .custom_hilbert import CustomHilbert

from fractions import Fraction

import numpy as _np
from netket import random as _random
from numba import jit

from typing import Optional, List


class Spin(CustomHilbert):
    r"""Hilbert space obtained as tensor product of local spin states."""

    def __init__(self, s: float, N: int = 1, total_sz: Optional[float] = None):
        r"""Hilbert space obtained as tensor product of local spin states.

        Args:
           s: Spin at each site. Must be integer or half-integer.
           N: Number of sites (default=1)
           total_sz: If given, constrains the total spin of system to a particular value.

        Examples:
           Simple spin hilbert space.

           >>> from netket.hilbert import Spin
           >>> g = Hypercube(length=10,n_dim=2,pbc=True)
           >>> hi = Spin(s=0.5, N=4)
           >>> print(hi.size)
           4
        """
        local_size = round(2 * s + 1)
        local_states = _np.empty(local_size)

        assert int(2 * s + 1) == local_size

        for i in range(local_size):
            local_states[i] = -round(2 * s) + 2 * i
        local_states = local_states.tolist()

        self._check_total_sz(total_sz, N)
        if total_sz is not None:

            def constraints(x):
                return self._sum_constraint(x, total_sz)

        else:
            constraints = None

        self._total_sz = total_sz if total_sz is None else int(total_sz)
        self._s = s
        self._local_size = local_size

        super().__init__(local_states, N, constraints)

    def random_state(self, out=None, rgen=None):
        r"""Member function generating uniformely distributed local random states.

        Args:
            out: If provided, the random quantum numbers will be inserted into this array.
                 It should be of the appropriate shape and dtype.
            rgen: The random number generator. If None, the global
                  NetKet random number generator is used.

        Examples:
           Test that a new random state is a possible state for the hilbert
           space.

           >>> import netket as nk
           >>> import numpy as np
           >>> hi = nk.hilbert.Boson(n_max=3, N=4)
           >>> rstate = np.zeros(hi.size)
           >>> hi.random_state(rstate)
           >>> local_states = hi.local_states
           >>> print(rstate[0] in local_states)
           True
        """

        if out is None:
            out = _np.empty(self._size)

        if rgen is None:
            rgen = _random

        if self._total_sz is None:
            for i in range(self._size):
                rs = rgen.randint(0, self._local_size)
                out[i] = self.local_states[rs]
        else:
            sites = list(range(self.size))

            out.fill(-round(2 * self._s))
            ss = self.size

            for i in range(round(self._s * self.size) + self._total_sz):
                s = rgen.randint(0, ss)

                out[sites[s]] += 2

                if out[sites[s]] > round(2 * self._s - 1):
                    sites.pop(s)
                    ss -= 1

        return out

    @staticmethod
    @jit(nopython=True)
    def _sum_constraint(x, total_sz):
        return _np.sum(x, axis=1) == round(2 * total_sz)

    def _check_total_sz(self, total_sz, size):
        if total_sz is None:
            return

        m = round(2 * total_sz)
        if _np.abs(m) > size:
            raise Exception(
                "Cannot fix the total magnetization: 2|M| cannot " "exceed Nspins."
            )

        if (size + m) % 2 != 0:
            raise Exception(
                "Cannot fix the total magnetization: Nspins + " "totalSz must be even."
            )

    def __pow__(self, n):
        if self._total_sz is None:
            total_sz = None
        else:
            total_sz = total_sz * n

        return Spin(self._s, self.size * n, total_sz=total_sz)

    def __repr__(self):
        total_sz = (
            ", total_sz={}".format(self._total_sz) if self._total_sz is not None else ""
        )
        return "Spin(s={}{}, N={})".format(Fraction(self._s), total_sz, self._size)
