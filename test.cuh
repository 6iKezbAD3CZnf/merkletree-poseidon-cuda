#ifndef TEST_CUH_
#define TEST_CUH_

#include "field.cuh"
#include "poseidon.cuh"

void check_test_vectors(std::vector<std::array<unsigned long long, SPONGE_WIDTH>> inputs, std::vector<std::array<unsigned long long, SPONGE_WIDTH>> expected_outputs);
void test_vectors();

#endif // TEST_CUH_
