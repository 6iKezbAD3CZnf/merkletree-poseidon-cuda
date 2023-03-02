#ifndef MERKLE_TREE_CUH_
#define MERKLE_TREE_CUH_

#include <cstdint>

#include "poseidon.cuh"

#define LEAVE_WIDTH 8
#define HASH_WIDTH 4
#define N_LEAVES (1 << 15)
#define N_DIGESTS (2 * N_LEAVES)

class MerkleTree {
public:
    MerkleTree(F leaves[N_LEAVES]);

    void fill_digests(F digests[N_DIGESTS], F leaves[N_LEAVES]);
    void print_digests();
    void print_root();

private:
    F leaves[LEAVE_WIDTH*N_LEAVES];
    F digests[HASH_WIDTH*N_DIGESTS];
};

#endif // MERKLE_TREE_CUH_
