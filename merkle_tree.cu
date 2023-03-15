#include <cassert>
#include <cmath>

#include "merkle_tree.cuh"

void host_fill_digest1(F digests[HASH_WIDTH*N_DIGESTS], F leaves[LEAVE_WIDTH*N_LEAVES], uint32_t left, uint32_t right, uint32_t to) {
    F state[SPONGE_WIDTH] = { F(0) };

    for (int k=0; k<SPONGE_WIDTH; k++) {
        if (k < HASH_WIDTH) {
            // left
            state[k] = digests[left*HASH_WIDTH + k];
        } else if (k < 2*HASH_WIDTH) {
            // right
            state[k] = digests[right*HASH_WIDTH + k - HASH_WIDTH];
        } else {
            state[k] = F(0);
        }
    }

    poseidon(state);

    for (int k=0; k<HASH_WIDTH; k++) {
        digests[to*HASH_WIDTH + k] = state[k];
    }
}

__device__
void device_fill_digests0_sub(F* d_digests, F* d_leaves, uint32_t from, uint32_t to) {
    F state[SPONGE_WIDTH];

    for (int k=0; k<SPONGE_WIDTH; k++) {
        if (k < LEAVE_WIDTH) {
            state[k] = d_leaves[from*LEAVE_WIDTH + k];
        } else {
            state[k] = F(0);
        }
    }

    poseidon(state);

    for (int k=0; k<HASH_WIDTH; k++) {
        d_digests[to*HASH_WIDTH + k] = state[k];
    }
}

__global__
void device_fill_digests0(F* d_digests, F* d_leaves) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < (N_LEAVES>>1)) {
        uint32_t from0 = i*2;
        uint32_t from1 = i*2 + 1;
        uint32_t to0 = i*4;
        uint32_t to1 = i*4 + 1;

        device_fill_digests0_sub(d_digests, d_leaves, from0, to0);
        device_fill_digests0_sub(d_digests, d_leaves, from1, to1);

        i += stride;
    }

    return;
}

__device__
void device_fill_digests1_sub(F* d_digests, F* d_leaves, uint32_t left, uint32_t right, uint32_t to) {
    F state[SPONGE_WIDTH];

    for (int k=0; k<SPONGE_WIDTH; k++) {
        if (k < HASH_WIDTH) {
            // left
            state[k] = d_digests[left*HASH_WIDTH + k];
        } else if (k < 2*HASH_WIDTH) {
            // right
            state[k] = d_digests[right*HASH_WIDTH + k - HASH_WIDTH];
        } else {
            state[k] = F(0);
        }
    }

    poseidon(state);

    for (int k=0; k<HASH_WIDTH; k++) {
        d_digests[to*HASH_WIDTH + k] = state[k];
    }
}

__global__
void device_fill_digests1(F* d_digests, F* d_leaves, int level, int n_level_leaves) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < n_level_leaves) {
        uint32_t level_start_idx = ((1<<level)-1)*2;
        uint32_t last_level_start_idx = ((1<<(level-1))-1)*2;

        uint32_t left0 = last_level_start_idx + ((1<<(level+1)) * (i*2));
        uint32_t right0 = left0 + 1;
        uint32_t left1 = last_level_start_idx + ((1<<(level+1)) * (i*2 + 1));
        uint32_t right1 = left1 + 1;
        uint32_t to0 = level_start_idx + (1<<(level+2)) * i;
        uint32_t to1 = to0 + 1;

        device_fill_digests1_sub(d_digests, d_leaves, left0, right0, to0);
        device_fill_digests1_sub(d_digests, d_leaves, left1, right1, to1);

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

    // int last_level_index = 0;
    // int level_index = N_LEAVES;
    // int n_level_leaves = N_LEAVES >> 1;
    int level = 1;
    int n_level_leaves = N_LEAVES >> 1;

    while (n_level_leaves > (1 << CAP_HEIGHT)) {
        device_fill_digests1<<<N_BLOCK, N_THREAD>>>(d_digests, d_leaves, level, n_level_leaves);
        cudaDeviceSynchronize();

        // last_level_index = level_index;
        // level_index += n_level_leaves;
        // n_level_leaves = n_level_leaves >> 1;
        level += 1;
        n_level_leaves = n_level_leaves >> 1;
    }

    cudaMemcpy(digests, d_digests, sizeof(F)*HASH_WIDTH*N_DIGESTS, cudaMemcpyDeviceToHost);

    cudaFree(d_leaves);
    cudaFree(d_digests);

    // caps
    uint32_t level_start_idx = ((1<<level)-1)*2;
    uint32_t last_level_start_idx = ((1<<(level-1))-1)*2;
    uint32_t left = last_level_start_idx;
    uint32_t right = left + 1;
    uint32_t to = level_start_idx;
    host_fill_digest1(digests, leaves, left, right, to);

    return;
}

void host_fill_digest0(F digests[HASH_WIDTH*N_DIGESTS], F leaves[LEAVE_WIDTH*N_LEAVES], uint32_t from, uint32_t to) {
    F state[SPONGE_WIDTH] = { F(0) };

    for (int k=0; k<SPONGE_WIDTH; k++) {
        if (k < LEAVE_WIDTH) {
            state[k] = leaves[from*LEAVE_WIDTH + k];
        } else {
            state[k] = F(0);
        }
    }

    poseidon(state);

    for (int k=0; k<HASH_WIDTH; k++) {
        digests[to*HASH_WIDTH + k] = state[k];
    }
}

void host_fill_digests(F digests[HASH_WIDTH*N_DIGESTS], F leaves[LEAVE_WIDTH*N_LEAVES]) {
    F state[SPONGE_WIDTH] = { F(0) };

    for (uint32_t i=0; i<(N_LEAVES>>1); i++) {
        uint32_t from0 = i*2;
        uint32_t from1 = i*2 + 1;
        uint32_t to0 = i*4;
        uint32_t to1 = i*4 + 1;
        host_fill_digest0(digests, leaves, from0, to0);
        host_fill_digest0(digests, leaves, from1, to1);
    }

    int level = 1;
    int n_level_leaves = N_LEAVES >> 1;

    while (n_level_leaves > (1 << CAP_HEIGHT)) {
        for (int i=0; i<(n_level_leaves>>1); i++) {
            uint32_t level_start_idx = ((1<<level)-1)*2;
            uint32_t last_level_start_idx = ((1<<(level-1))-1)*2;

            uint32_t left0 = last_level_start_idx + ((1<<(level+1)) * (i*2));
            uint32_t right0 = left0 + 1;
            uint32_t left1 = last_level_start_idx + ((1<<(level+1)) * (i*2 + 1));
            uint32_t right1 = left1 + 1;
            uint32_t to0 = level_start_idx + (1<<(level+2)) * i;
            uint32_t to1 = to0 + 1;

            host_fill_digest1(digests, leaves, left0, right0, to0);
            host_fill_digest1(digests, leaves, left1, right1, to1);
        }

        level += 1;
        n_level_leaves = n_level_leaves >> 1;
    }

    // caps
    uint32_t level_start_idx = ((1<<level)-1)*2;
    uint32_t last_level_start_idx = ((1<<(level-1))-1)*2;
    uint32_t left = last_level_start_idx;
    uint32_t right = left + 1;
    uint32_t to = level_start_idx;
    host_fill_digest1(digests, leaves, left, right, to);

    return;
}

MerkleTree::MerkleTree(bool is_host, F leaves[LEAVE_WIDTH*N_LEAVES], uint32_t cap_height) {
    assert(cap_height == CAP_HEIGHT);

    if (is_host) {
        host_fill_digests(digests, leaves);
    } else {
        device_fill_digests(digests, leaves);
    }
}

void MerkleTree::print_leaves() {
    for (int i=0; i<N_LEAVES; i++) {
        std::cout << std::dec;
        std::cout << "leave" << i << " is [";
        std::cout << std::hex;
        for (int j=0; j<LEAVE_WIDTH; j++) {
            std::cout << leaves[i*LEAVE_WIDTH + j] << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::dec;
}

void MerkleTree::print_digests() {
    for (int i=0; i<N_DIGESTS; i++) {
        std::cout << std::dec;
        std::cout << "digest" << i << " is [";
        std::cout << std::hex;
        for (int j=0; j<HASH_WIDTH; j++) {
            std::cout << digests[i*HASH_WIDTH + j] << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::dec;
}

void MerkleTree::print_root() {
    std::cout << std::hex;
    for (int j=0; j<HASH_WIDTH; j++) {
        std::cout << digests[(N_DIGESTS - 1)*HASH_WIDTH + j] << ", ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::dec;
}
