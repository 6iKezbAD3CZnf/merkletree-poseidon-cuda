#include <cassert>

#include "merkle_tree.cuh"

extern "C"
F* test_cc(F* input) {
    F* ret = (F*)malloc(sizeof(F)*3);
    ret[0] = input[0];
    ret[1] = input[1];
    ret[2] = input[2];
    return ret;
}

extern "C"
F* fill_digests(F* leaves, uint32_t leaves_len, uint32_t cap_height) {
    assert(leaves_len == N_LEAVES);
    assert(cap_height == CAP_HEIGHT);

    F* digests = (F*)malloc(sizeof(F)*HASH_WIDTH*N_DIGESTS);
    host_fill_digests(digests, leaves);

    return digests;
}
