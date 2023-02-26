#ifndef POSEIDON_CUH_
#define POSEIDON_CUH_

#include <array>
#include <vector>

#include "field.cuh"

std::array<F, SPONGE_WIDTH> poseidon(std::array<F, SPONGE_WIDTH> input);

#endif // POSEIDON_CUH_
