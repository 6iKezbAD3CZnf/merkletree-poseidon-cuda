#ifndef MERKLE_TREE_CUH_
#define MERKLE_TREE_CUH_

#include <cstdint>

#include "poseidon.cuh"

#define N_BLOCK 8
#define N_THREAD 576

#define LEAVE_WIDTH 8
#define HASH_WIDTH 4
#define CAP_HEIGHT 0
#define N_LEAVES (1 << 3)
#define N_DIGESTS (2 * (N_LEAVES - (1 << CAP_HEIGHT)) + 1)

void host_fill_digests(F digests[HASH_WIDTH*N_DIGESTS], F leaves[LEAVE_WIDTH*N_LEAVES]);

class MerkleTree {
public:
    MerkleTree(bool is_host, F leaves[N_LEAVES], uint32_t cap_height);

    void print_leaves();
    void print_digests();
    void print_root();

// private:
    F leaves[LEAVE_WIDTH*N_LEAVES];
    F digests[HASH_WIDTH*N_DIGESTS];
};

#endif // MERKLE_TREE_CUH_
