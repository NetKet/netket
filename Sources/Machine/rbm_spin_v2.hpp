// Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#ifndef SOURCES_MACHINE_RBM_SPIN_V2_HPP
#define SOURCES_MACHINE_RBM_SPIN_V2_HPP

#include <cmath>
#include <memory>

#include <Eigen/Core>
#include <nonstd/optional.hpp>

#include "Hilbert/abstract_hilbert.hpp"
#include "Machine/abstract_machine.hpp"
#include "common_types.hpp"

namespace netket {

class RbmSpinV2 : public AbstractMachine {
 public:
  RbmSpinV2(std::shared_ptr<const AbstractHilbert> hilbert, Index nhidden,
            Index alpha, bool usea, bool useb, Index const batch_size);

  int Npar() const final {
    return W_.size() + (a_.has_value() ? a_->size() : 0) +
           (b_.has_value() ? b_->size() : 0);
  }
  int Nvisible() const final { return W_.rows(); }
  int Nhidden() const noexcept { return W_.cols(); }

  /// Returns current batch size.
  Index BatchSize() const noexcept;

  /// \brief Updates the batch size.
  ///
  /// There is no need to call this function explicitly -- batch size is changed
  /// automatically on calls to `LogVal` and `DerLog`.
  void BatchSize(Index batch_size);

  VectorType GetParameters() final;
  void SetParameters(Eigen::Ref<const Eigen::VectorXcd> pars) final;

  void LogVal(Eigen::Ref<const RowMatrix<double>> v,
              Eigen::Ref<Eigen::VectorXcd> out, const any & /*unused*/) final;

  void DerLog(Eigen::Ref<const RowMatrix<double>> v,
              Eigen::Ref<RowMatrix<Complex>> out, const any & /*unused*/) final;

  PyObject *StateDict() final;

  bool IsHolomorphic() const noexcept final { return true; }

  NETKET_MACHINE_DISABLE_LOOKUP

 private:
  /// Performs `out := log(cosh(out + b))`.
  void ApplyBiasAndActivation(Eigen::Ref<Eigen::VectorXcd> out) const;

  Eigen::MatrixXcd W_;             ///< weights
  nonstd::optional<VectorXcd> a_;  ///< visible units bias
  nonstd::optional<VectorXcd> b_;  ///< hidden units bias

  /// Caches
  RowMatrix<Complex> theta_;
};

}  // namespace netket

#endif  // SOURCES_MACHINE_RBM_SPIN_V2_HPP
