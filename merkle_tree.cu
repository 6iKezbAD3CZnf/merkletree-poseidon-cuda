#include <cassert>
#include <cmath>

#include "merkle_tree.cuh"

MerkleTree::MerkleTree(F leaves[LEAVE_WIDTH*N_LEAVES]) {
    fill_digests(digests, leaves);
}

void MerkleTree::fill_digests(F digests[HASH_WIDTH*N_DIGESTS], F leaves[LEAVE_WIDTH*N_LEAVES]) {
    F state[SPONGE_WIDTH] = { F(0) };

    for (int j=0; j<N_LEAVES; j++) {
        for (int k=0; k<SPONGE_WIDTH; k++) {
            if (k < LEAVE_WIDTH) {
                state[k] = leaves[j*LEAVE_WIDTH + k];
            } else {
                state[k] = F(0);
            }
        }

        poseidon(state);

        for (int k=0; k<HASH_WIDTH; k++) {
            digests[j*HASH_WIDTH + k] = state[k];
        }
    }

    int last_level_index = 0;
    int level_index = N_LEAVES;
    int n_level_leaves = N_LEAVES >> 1;

    while (n_level_leaves > 0) {
        for (int j=0; j<n_level_leaves; j++) {
            for (int k=0; k<SPONGE_WIDTH; k++) {
                if (k < HASH_WIDTH) {
                    // left
                    state[k] = digests[(last_level_index + j*2)*HASH_WIDTH + k];
                } else if (k < 2*HASH_WIDTH) {
                    // right
                    state[k] = digests[(last_level_index + (j*2+1))*HASH_WIDTH + k - HASH_WIDTH];
                } else {
                    state[k] = F(0);
                }
            }

            poseidon(state);

            for (int k=0; k<HASH_WIDTH; k++) {
                digests[(level_index + j)*HASH_WIDTH + k] = state[k];
            }
        }

        last_level_index = level_index;
        level_index += n_level_leaves;
        n_level_leaves = n_level_leaves >> 1;
    }

    return;
}

void MerkleTree::print_digests() {
    std::cout << std::hex;
    for (int i=0; i<N_DIGESTS; i++) {
        for (int j=0; j<HASH_WIDTH; j++) {
            std::cout << digests[i*HASH_WIDTH + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::dec;
}

void MerkleTree::print_root() {
    std::cout << std::hex;
    for (int j=0; j<HASH_WIDTH; j++) {
        std::cout << digests[(N_DIGESTS - 2)*HASH_WIDTH + j] << ", ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::dec;
}
