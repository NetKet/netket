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

#ifndef NETKET_PYADADELTA_HPP
#define NETKET_PYADADELTA_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "ada_delta.hpp"

namespace py = pybind11;

namespace netket {

void AddAdaDelta(py::module &subm) {
  py::class_<AdaDelta, AbstractOptimizer>(subm, "AdaDelta")
      .def(py::init<double, double>(), py::arg("rho") = 0.95,
           py::arg("epscut") = 1.0e-7);
}

}  // namespace netket

#endif
