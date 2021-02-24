import numpy as np
from numba import jit

from netket.graph import AbstractGraph, Graph
from netket.hilbert import (
    AbstractHilbert,
)

from ._abstract_operator import (
    AbstractOperator,
)
from ._graph_operator import (
    GraphOperator,
)

from . import spin
from ._local_operator import LocalOperator


class SpecialHamiltonian(AbstractOperator):
    def to_local_operator(self):
        raise NotImplementedError(
            "Must implemented to_local_operator for {}".format(type(self))
        )

    def conjugate(self, *, concrete=True):
        return self.to_local_operator().conjugate(concrete=concrete)

    def __add__(self, other):
        if type(self) is type(other):
            res = self.copy()
            res += other
            return res

        return self.to_local_operator().__add__(other)

    def __sub__(self, other):
        if type(self) is type(other):
            res = self.copy()
            res -= other
            return res

        return self.to_local_operator().__sub__(other)

    def __radd__(self, other):
        if type(self) is type(other):
            res = self.copy()
            res += other
            return res

        return self.to_local_operator().__radd__(other)

    def __rsub__(self, other):
        if type(self) is type(other):
            res = self.copy()
            res -= other
            return res

        return self.to_local_operator().__rsub__(other)

    def __iadd__(self, other):
        if type(self) is type(other):
            self._iadd_same_hamiltonian(other)
            return self

        return NotImplemented

    def __isub__(self, other):
        if type(self) is type(other):
            self._isub_same_hamiltonian(other)
            return self

        return NotImplemented

    def __mul__(self, other):
        return self.to_local_operator().__mul__(other)

    def __rmul__(self, other):
        return self.to_local_operator().__rmul__(other)

    def __matmul__(self, other):
        return self.to_local_operator().__matmul__(other)


class Ising(SpecialHamiltonian):
    def __init__(
        self,
        hilbert: AbstractHilbert,
        graph: AbstractGraph,
        h: float,
        J: float = 1.0,
        dtype=float,
    ):
        r"""
        Constructs a new ``Ising`` given a hilbert space, a transverse field,
        and (if specified) a coupling constant.

        Args:
            hilbert: Hilbert space the operator acts on.
            h: The strength of the transverse field.
            J: The strength of the coupling. Default is 1.0.

        Examples:
            Constructs an ``Ising`` operator for a 1D system.

            >>> import netket as nk
            >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
            >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
            >>> op = nk.operator.Ising(h=1.321, hilbert=hi, J=0.5)
            >>> print(op.hilbert.size)
            20
        """
        assert (
            graph.n_nodes == hilbert.size
        ), "The size of the graph must match the hilbert space"

        super().__init__(hilbert)

        self._h = h
        self._J = J
        self._edges = np.asarray(list(graph.edges()))

        self._dtype = dtype

    @property
    def h(self):
        return self._h

    @property
    def J(self):
        return self._J

    @property
    def edges(self):
        return self._edges

    @property
    def is_hermitian(self):
        return True

    @property
    def dtype(self):
        return self._dtype

    def conjugate(self, *, concrete=True):
        # if real
        if isinstance(self.h, float) and isinstance(self.J, float):
            return self
        else:
            raise NotImplementedError

    @staticmethod
    @jit(nopython=True)
    def n_conn(x, out):
        r"""Return the number of states connected to x.

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            out (array): If None an output array is allocated.

        Returns:
            array: The number of connected states x' for each x[i].

        """
        if out is None:
            out = np.empty(
                x.shape[0],
                dtype=np.int32,
            )

        out.fill(x.shape[1] + 1)

        return out

    def copy(self):
        graph = Graph(edges=[list(edge) for edge in self.edges])
        return Ising(hilbert=self.hilbert, graph=graph, J=self.J, h=self.h)

    def to_local_operator(self):
        # The hamiltonian
        ha = LocalOperator(self.hilbert, dtype=self.dtype)

        if self.h != 0:
            for i in range(self.hilbert.size):
                ha -= self.h * spin.sigmax(self.hilbert, i)

        if self.J != 0:
            for (i, j) in self.edges:
                ha += self.J * (
                    spin.sigmaz(self.hilbert, i) * spin.sigmaz(self.hilbert, j)
                )

        return ha

    def _iadd_same_hamiltonian(self, other):
        if self.hilbert != other.hilbert:
            raise NotImplementedError(
                "Cannot add hamiltonians on different hilbert spaces"
            )

        self._h += other.h
        self._J += other.J

    def _isub_same_hamiltonian(self, other):
        if self.hilbert != other.hilbert:
            raise NotImplementedError(
                "Cannot add hamiltonians on different hilbert spaces"
            )

        self._h -= other.h
        self._J -= other.J

    def get_conn(self, x):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x (array): An array of shape (hilbert.size) containing the quantum numbers x.

        Returns:
            matrix: The connected states x' of shape (N_connected,hilbert.size)
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """
        return self._flattened_kernel(
            x.reshape((1, -1)),
            np.ones(1),
            self._edges,
            self._h,
            self._J,
        )

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(
        x,
        sections,
        edges,
        h,
        J,
    ):
        n_sites = x.shape[1]
        n_conn = n_sites + 1

        x_prime = np.empty(
            (
                x.shape[0] * n_conn,
                n_sites,
            )
        )
        mels = np.empty(x.shape[0] * n_conn)

        diag_ind = 0

        for i in range(x.shape[0]):
            mels[diag_ind] = 0.0
            for k in range(edges.shape[0]):
                mels[diag_ind] += (
                    J
                    * x[
                        i,
                        edges[
                            k,
                            0,
                        ],
                    ]
                    * x[
                        i,
                        edges[
                            k,
                            1,
                        ],
                    ]
                )

            odiag_ind = 1 + diag_ind

            mels[odiag_ind : (odiag_ind + n_sites)].fill(-h)

            x_prime[diag_ind : (diag_ind + n_conn)] = np.copy(x[i])

            for j in range(n_sites):
                x_prime[j + odiag_ind][j] *= -1.0

            diag_ind += n_conn

            sections[i] = diag_ind

        return x_prime, mels

    def get_conn_flattened(
        self,
        x,
        sections,
        pad=False,
    ):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            sections (array): An array of size (batch_size) useful to unflatten
                        the output of this function.
                        See numpy.split for the meaning of sections.
            pad (bool): no effect here

        Returns:
            matrix: The connected states x', flattened together in a single matrix.
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        return self._flattened_kernel(
            x,
            sections,
            self._edges,
            self._h,
            self._J,
        )

    def __repr__(self):
        return f"Ising(J={self._J}, h={self._h}; dim={self.hilbert.size})"


class Heisenberg(GraphOperator):
    def __init__(
        self,
        hilbert,
        graph,
        J=1,
        sign_rule=None,
    ):
        """
        Constructs a new ``Heisenberg`` given a hilbert space.

        Args:
            hilbert: Hilbert space the operator acts on.
            grah: The graph upon which this hamiltonian is defined.
            J: The strength of the coupling. Default is 1.
            sign_rule: If enabled, Marshal's sign rule will be used. On a bipartite
                       lattice, this corresponds to a basis change flipping the Sz direction
                       at every odd site of the lattice. For non-bipartite lattices, the
                       sign rule cannot be applied. Defaults to True if the lattice is
                       bipartite, False otherwise.

        Examples:
         Constructs a ``Heisenberg`` operator for a 1D system.

            >>> import netket as nk
            >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
            >>> hi = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g)
            >>> op = nk.operator.Heisenberg(hilbert=hi)
            >>> print(op.hilbert.size)
            20
        """
        if sign_rule is None:
            sign_rule = graph.is_bipartite()

        self._J = J
        self._sign_rule = sign_rule

        sz_sz = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )
        exchange = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 2, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        if sign_rule:
            if not graph.is_bipartite():
                raise ValueError("sign_rule=True specified for a non-bipartite lattice")
            heis_term = sz_sz - exchange
        else:
            heis_term = sz_sz + exchange

        super().__init__(
            hilbert,
            graph,
            bond_ops=[J * heis_term],
        )

    @property
    def J(self):
        return self._J

    @property
    def is_hermitian(self):
        return True

    @property
    def uses_sign_rule(self):
        return self._sign_rule

    def __repr__(self):
        return f"Heisenberg(J={self._J}, sign_rule={self._sign_rule}; dim={self.hilbert.size})"
