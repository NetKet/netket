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

#ifndef NETKET_GRAPH_HPP
#define NETKET_GRAPH_HPP
#include <array>
#include <unordered_map>
#include <vector>
#include "Utils/json_utils.hpp"
#include "abstract_graph.hpp"
#include "custom_graph.hpp"
#include "hypercube.hpp"

namespace netket {

class Graph : public AbstractGraph {
  using Ptype = std::unique_ptr<AbstractGraph>;
  Ptype g_;

 public:
  explicit Graph(const json& pars) {
    // Check if a graph is explicitely defined in the input
    if (FieldExists(pars, "Graph")) {
      // Checking if we are using a graph in the hard-coded library
      if (FieldExists(pars["Graph"], "Name")) {
        std::string graph_name = pars["Graph"]["Name"];
        if (graph_name == "Hypercube") {
          g_ = Ptype(new Hypercube(pars));
        } else {
          std::stringstream s;
          s << "Unknown Graph type: " << graph_name;
          throw InvalidInputError(s.str());
        }
      }
      // Otherwise using a user-defined graph
      else {
        g_ = Ptype(new CustomGraph(pars));
      }
    } else {
      // Otherwise try to construct a custom graph using Hilbert space
      // information
      g_ = Ptype(new CustomGraph(pars));
    }
  }

  int Nsites() const override { return g_->Nsites(); }

  std::vector<std::vector<int>> AdjacencyList() const override {
    return g_->AdjacencyList();
  }

  std::vector<std::vector<int>> SymmetryTable() const override {
    return g_->SymmetryTable();
  }

  std::vector<std::vector<int>> Distances() const override {
    return g_->Distances();
  }

  const ColorMap& EdgeColors() const override { return g_->EdgeColors(); }

  template<typename Func>
  void BreadthFirstSearch(int start, int max_depth, Func visitor_func) const {
    g_->BreadthFirstSearch(start, max_depth, visitor_func);
  }

  template<typename Func>
  void BreadthFirstSearch(int start, Func visitor_func) const {
    BreadthFirstSearch(start, Nsites(), visitor_func);
  }

  template<typename Func>
  void FullBreadthFirstSearch(Func visitor_func) const {
    g_->FullBreadthFirstSearch(visitor_func);
  }

  bool IsBipartite() const override { return g_->IsBipartite(); }

  bool IsConnected() const override { return g_->IsConnected(); }

};
}  // namespace netket

#endif
