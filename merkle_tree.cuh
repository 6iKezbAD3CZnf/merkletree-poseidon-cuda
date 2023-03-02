#ifndef MERKLE_TREE_CUH_
#define MERKLE_TREE_CUH_

#include <cstdint>

#include "poseidon.cuh"

#define N_BLOCK 8
#define N_THREAD 500

#define LEAVE_WIDTH 8
#define HASH_WIDTH 4
#define N_LEAVES (1 << 19)
#define N_DIGESTS (2 * N_LEAVES)

class MerkleTree {
public:
    MerkleTree(bool is_host, F leaves[N_LEAVES]);

    void host_fill_digests(F digests[N_DIGESTS], F leaves[N_LEAVES]);
    void print_digests();
    void print_root();

private:
    F leaves[LEAVE_WIDTH*N_LEAVES];
    F digests[HASH_WIDTH*N_DIGESTS];
};

#endif // MERKLE_TREE_CUH_
