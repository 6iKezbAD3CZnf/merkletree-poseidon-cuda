#include "poseidon.cuh"

// Poseidon Hash
std::array<F, SPONGE_WIDTH> poseidon(std::array<F, SPONGE_WIDTH> input) {
    std::array<F, SPONGE_WIDTH> output;
    for (int i=0; i<SPONGE_WIDTH; i++) {
        output[i] = F(input[i]);
    }
    return output;
}
