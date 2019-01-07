// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_PYHILBERT_HPP
#define NETKET_PYHILBERT_HPP

#include "hilbert.hpp"
#include <complex>
#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;

namespace netket {

constexpr int HilbertIndex::MaxStates;

#define ADDHILBERTMETHODS(name)                                                       \
                                                                                      \
  .def_property_readonly(                                                             \
      "is_discrete", &name::IsDiscrete,                                               \
      R"EOF(bool: Whether the hilbert space is discrete.)EOF")                        \
      .def_property_readonly("local_size", &name::LocalSize,                          \
                             R"EOF(int: Size of the local hilbert space.)EOF")        \
      .def_property_readonly(                                                         \
          "size", &name::Size,                                                        \
          R"EOF(int: The number of visible units needed to describe the system.)EOF") \
      .def_property_readonly(                                                         \
          "local_states", &name::LocalStates,                                         \
          R"EOF(list[float]: List of discreet local quantum numbers.)EOF")            \
      .def("random_vals", &name ::RandomVals)                                         \
      .def("update_conf", &name::UpdateConf)

void AddHilbertModule(py::module &m) {
  auto subm = m.def_submodule("hilbert");

  py::class_<AbstractHilbert>(subm, "Hilbert")
      ADDHILBERTMETHODS(AbstractHilbert);

  py::class_<Spin, AbstractHilbert>(
      subm, "Spin", R"EOF(Hilbert space composed of spin states.)EOF")
      .def(py::init<const AbstractGraph &, double>(), py::keep_alive<1, 2>(),
           py::arg("graph"), py::arg("s"), R"EOF(
           Constructs a new ``Spin`` given a graph and the value of each spin.

           Args:
               graph: Graph representation of sites.
               s: Spin at each site. Must be integer or half-integer.

           Examples:
               Simple spin hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Spin
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Spin(graph=g, s=0.5)
               >>> print(hi.size)
               100

               ```
           )EOF")
      .def(py::init<const AbstractGraph &, double, double>(),
           py::keep_alive<1, 2>(), py::arg("graph"), py::arg("s"),
           py::arg("total_sz"), R"EOF(
           Constructs a new ``Spin`` given a graph and the value of each spin.

           Args:
               graph: Graph representation of sites.
               s: Spin at each site. Must be integer or half-integer.
               total_sz: Constrain total spin of system to a particular value.

           Examples:
               Simple spin hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Spin
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Spin(graph=g, s=0.5, total_sz=0)
               >>> print(hi.size)
               100

               ```
           )EOF") ADDHILBERTMETHODS(Spin);

  py::class_<Qubit, AbstractHilbert>(
      subm, "Qubit", R"EOF(Hilbert space composed of qubits.)EOF")
      .def(py::init<const AbstractGraph &>(), py::keep_alive<1, 2>(),
           py::arg("graph"), R"EOF(
           Constructs a new ``Qubit`` given a graph.

           Args:
               graph: Graph representation of sites.

           Examples:
               Simple qubit hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Qubit
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Qubit(graph=g)
               >>> print(hi.size)
               100

               ```
           )EOF") ADDHILBERTMETHODS(Qubit);

  py::class_<Boson, AbstractHilbert>(
      subm, "Boson", R"EOF(Hilbert space composed of bosonic states.)EOF")
      .def(py::init<const AbstractGraph &, int>(), py::keep_alive<1, 2>(),
           py::arg("graph"), py::arg("n_max"), R"EOF(
           Constructs a new ``Boson`` given a graph and maximum occupation number.

           Args:
               graph: Graph representation of sites.
               n_max: Maximum occupation for a site.

           Examples:
               Simple boson hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Boson
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Boson(graph=g, n_max=4)
               >>> print(hi.size)
               100

               ```
           )EOF")
      .def(py::init<const AbstractGraph &, int, int>(), py::keep_alive<1, 2>(),
           py::arg("graph"), py::arg("n_max"), py::arg("n_bosons"), R"EOF(
           Constructs a new ``Boson`` given a graph,  maximum occupation number,
           and total number of bosons.

           Args:
               graph: Graph representation of sites.
               n_max: Maximum occupation for a site.
               n_bosons: Constraint for the number of bosons.

           Examples:
               Simple boson hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Boson
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Boson(graph=g, n_max=5, n_bosons=11)
               >>> print(hi.size)
               100

               ```
           )EOF") ADDHILBERTMETHODS(Boson);

  py::class_<CustomHilbert, AbstractHilbert>(subm, "CustomHilbert",
                                             R"EOF(A custom hilbert space.)EOF")
      .def(py::init<const AbstractGraph &, std::vector<double>>(),
           py::keep_alive<1, 2>(), py::arg("graph"), py::arg("local_states"),
           R"EOF(
           Constructs a new ``CustomHilbert`` given a graph and a list of 
           eigenvalues of the states.

           Args:
               graph: Graph representation of sites.
               local_states: Eigenvalues of the states.

           Examples:
               Simple custom hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import CustomHilbert
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = CustomHilbert(graph=g, local_states=[-1232, 132, 0])
               >>> print(hi.size)
               100

               ```
           )EOF") ADDHILBERTMETHODS(CustomHilbert);

  py::class_<HilbertIndex>(subm, "HilbertIndex")
      .def(py::init<const AbstractHilbert &>(), py::arg("hilbert"))
      .def_property_readonly("n_states", &HilbertIndex::NStates)
      .def("number_to_state", &HilbertIndex::NumberToState)
      .def("state_to_number", &HilbertIndex::StateToNumber)
      .def_readonly_static("max_states", &HilbertIndex::MaxStates);

} // namespace netket

} // namespace netket

#endif
