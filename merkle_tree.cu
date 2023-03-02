#include <cassert>
#include <cmath>

#include "merkle_tree.cuh"

__global__
void device_fill_digests0(F* d_digests, F* d_leaves) {
    F state[SPONGE_WIDTH];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < N_LEAVES) {
        for (int k=0; k<SPONGE_WIDTH; k++) {
            if (k < LEAVE_WIDTH) {
                state[k] = d_leaves[i*LEAVE_WIDTH + k];
            } else {
                state[k] = F(0);
            }
        }

        poseidon(state);

        for (int k=0; k<HASH_WIDTH; k++) {
            d_digests[i*HASH_WIDTH + k] = state[k];
        }

        i += stride;
    }

    return;
}

__global__
void device_fill_digests1(F* d_digests, F* d_leaves, int last_level_index, int level_index, int n_level_leaves) {
    F state[SPONGE_WIDTH];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < n_level_leaves) {
        for (int k=0; k<SPONGE_WIDTH; k++) {
            if (k < HASH_WIDTH) {
                // left
                state[k] = d_digests[(last_level_index + i*2)*HASH_WIDTH + k];
            } else if (k < 2*HASH_WIDTH) {
                // right
                state[k] = d_digests[(last_level_index + (i*2+1))*HASH_WIDTH + k - HASH_WIDTH];
            } else {
                state[k] = F(0);
            }
        }

        poseidon(state);

        for (int k=0; k<HASH_WIDTH; k++) {
            d_digests[(level_index + i)*HASH_WIDTH + k] = state[k];
        }

        i += stride;
    }

    return;
}

void device_fill_digests(F digests[HASH_WIDTH*N_DIGESTS], F leaves[LEAVE_WIDTH*N_LEAVES]) {
    F* d_digests;
    F* d_leaves;
    cudaMalloc(&d_leaves, sizeof(F)*LEAVE_WIDTH*N_DIGESTS);
    cudaMalloc(&d_digests, sizeof(F)*HASH_WIDTH*N_DIGESTS);
    cudaMemcpy(d_leaves, leaves, sizeof(F)*LEAVE_WIDTH*N_LEAVES, cudaMemcpyHostToDevice);

    device_fill_digests0<<<N_BLOCK, N_THREAD>>>(d_digests, d_leaves);
    cudaDeviceSynchronize();

    int last_level_index = 0;
    int level_index = N_LEAVES;
    int n_level_leaves = N_LEAVES >> 1;

    while (n_level_leaves > 0) {
        device_fill_digests1<<<N_BLOCK, N_THREAD>>>(d_digests, d_leaves, last_level_index, level_index, n_level_leaves);
        cudaDeviceSynchronize();

        last_level_index = level_index;
        level_index += n_level_leaves;
        n_level_leaves = n_level_leaves >> 1;
    }

    cudaMemcpy(digests, d_digests, sizeof(F)*HASH_WIDTH*N_DIGESTS, cudaMemcpyDeviceToHost);

    cudaFree(d_leaves);
    cudaFree(d_digests);
}

MerkleTree::MerkleTree(bool is_host, F leaves[LEAVE_WIDTH*N_LEAVES]) {
    if (is_host) {
        host_fill_digests(digests, leaves);
    } else {
        device_fill_digests(digests, leaves);
    }
}

void MerkleTree::host_fill_digests(F digests[HASH_WIDTH*N_DIGESTS], F leaves[LEAVE_WIDTH*N_LEAVES]) {
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
