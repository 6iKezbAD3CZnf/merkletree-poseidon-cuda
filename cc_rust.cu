#include <cassert>

#include "merkle_tree.cuh"

extern "C"
F* digests_cap(F* leaves, uint32_t leaves_len, uint32_t cap_height) {
    assert(leaves_len == N_LEAVES);
    assert(cap_height == CAP_HEIGHT);

    F* digests_cap = (F*)malloc(sizeof(F)*HASH_WIDTH*N_DIGESTS);
    // host_fill_digests_cap(digests_cap, leaves);
    device_fill_digests_cap(digests_cap, leaves);

    return digests_cap;
}
