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

#include "catch.hpp"
#include "netket.hpp"
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "graph_input_tests.hpp"

TEST_CASE("graphs have consistent number of sites", "[graph]") {

  auto input_tests = GetGraphInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t i = 0; i < ntests; i++) {
    std::string name = input_tests[i].dump();

    SECTION("Graph test (" + std::to_string(i) + ") on " + name) {

      netket::Graph graph(input_tests[i]);

      REQUIRE(graph.Nsites() > 0);
    }
  }
}

TEST_CASE("Breadth-first search", "[graph]") {
  const auto input_tests = GetGraphInputs();

  for(const auto input : input_tests) {
    const auto data = input.dump();
    SECTION("each node is visited at most once on " + data) {
      netket::Graph graph(input);
      for(int start = 0; start < graph.Nsites(); ++start) {
        std::unordered_set<int> visited;
        int ncall = 0;
        graph.BreadthFirstSearch(start, [&](int v, int depth) {
          INFO("ncall: " << ncall << ", start: " << start << ", v: " << v << ", depth: " << depth);
          REQUIRE(visited.count(v) == 0);
          visited.insert(v);
          ncall++;
        });
      }
    }
  }

  SECTION("each node is visited at least once on connected graph") {
    netket::json pars;
    pars = {
        {"Graph",
         {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", false}}}};
    netket::Graph graph(pars);
    for(int i = 0; i < graph.Nsites(); ++i) {
      std::unordered_set<int> visited;
      graph.BreadthFirstSearch(i, [&visited](int v, int /*depth*/) {
        visited.insert(v);
      });
      for(int v = 0; v < graph.Nsites(); ++v) {
        INFO("v: " << v);
        REQUIRE(visited.count(v) == 1);
      }
    }
  }
}
