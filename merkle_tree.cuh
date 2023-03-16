#ifndef MERKLE_TREE_CUH_
#define MERKLE_TREE_CUH_

#include <cstdint>

#include "poseidon.cuh"

#define N_BLOCK 8
#define N_THREAD 576

#define LEAVE_WIDTH 8
#define HASH_WIDTH 4

void host_fill_digests_cap(
        F* digests_cap,
        F* leaves,
        uint32_t n_leaves,
        uint32_t cap_height
        );
void device_fill_digests_cap(
        F* digests_cap,
        uint32_t n_digests_cap,
        F* leaves,
        uint32_t n_leaves,
        uint32_t cap_height
        );

void print_leaves(F* leaves, uint32_t n_leaves);
void print_digests(F* digests, uint32_t n_digests);
void print_cap(F* digests_cap, uint32_t n_digests, uint32_t cap_height);

#endif // MERKLE_TREE_CUH_
