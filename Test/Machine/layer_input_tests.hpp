
#include <fstream>
#include <string>
#include <vector>
#include "Utils/json_utils.hpp"

std::vector<netket::json> GetLayerInputs() {
  std::vector<netket::json> input_tests;
  netket::json pars;

  // FullyConnected layer
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine",
           {{"Name", "FFNN"},
            {"Layers",
             {{{"Name", "FullyConnected"},
               {"Inputs", 20},
               {"Outputs", 40},
               {"Activation", "Lncosh"}}}}}},
          {"Hamiltonian", {{"Name", "Heisenberg"}}}};
  input_tests.push_back(pars);

  // Convolutional layer
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine",
           {{"Name", "FFNN"},
            {"Layers",
             {{{"Name", "Convolutional"},
               {"InputChannels", 1},
               {"OutputChannels", 2},
               {"Distance", 2},
               {"Activation", "Lncosh"}},
              {{"Name", "Convolutional"},
               {"InputChannels", 2},
               {"OutputChannels", 2},
               {"Distance", 2},
               {"Activation", "Lncosh"}}}}}},
          {"Hamiltonian", {{"Name", "Heisenberg"}}}};
  input_tests.push_back(pars);

  return input_tests;
}
