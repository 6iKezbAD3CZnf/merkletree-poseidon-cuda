#ifndef MERKLE_TREE_CUH_
#define MERKLE_TREE_CUH_

#include <cstdint>

#include "poseidon.cuh"

#define N_BLOCK 8
#define N_THREAD 576

#define SPONGE_RATE 8
#define HASH_WIDTH 4

void host_fill_digests_caps(
        F* digests_caps,
        uint32_t num_digests,
        F* leaves,
        uint32_t num_leaves,
        uint32_t leave_len,
        uint32_t cap_height
        );
void device_fill_digests_caps(
        F* digests_caps,
        uint32_t n_digests_cap,
        F* leaves,
        uint32_t num_leaves,
        uint32_t leave_len,
        uint32_t cap_height
        );

void print_leaves(F* leaves, uint32_t num_leaves, uint32_t leave_len);
void print_digests(F* digests, uint32_t num_digests);
void print_caps(F* digests_caps, uint32_t num_digests, uint32_t cap_height);

#endif // MERKLE_TREE_CUH_
