#include <cassert>
#include <cmath>

#include "merkle_tree.cuh"

__host__ __device__
void two_to_one(F* digests, uint32_t left, uint32_t right, uint32_t to) {
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

__host__ __device__
void permute(F* digests, F* leaves, uint32_t from, uint32_t to) {
    F state[SPONGE_WIDTH];

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

__global__
void device_fill_digests0(F* d_digests, F* d_leaves, uint32_t n_leaves) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < n_leaves) {
        uint32_t from = i;
        uint32_t to = (i>>1<<2) | (i&0b1);
        permute(d_digests, d_leaves, from, to);
        i += stride;
    }
}

__global__
void device_fill_digests1(
        F* d_digests,
        uint32_t level,
        uint32_t n_level_leaves,
        uint32_t last_level_start_idx,
        uint32_t level_start_idx
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < n_level_leaves) {
        uint32_t left = last_level_start_idx + i*(1<<(level+1));
        uint32_t right = left + 1;
        uint32_t to = (level_start_idx + (i>>1)*(1<<(level+2))) | (i&0b1);
        two_to_one(d_digests, left, right, to);

        i += stride;
    }

    return;
}

void device_fill_digests_cap(F* digests_cap, uint32_t n_digests_cap, F* leaves, uint32_t n_leaves, uint32_t cap_height) {
    F* d_digests;
    F* d_leaves;
    cudaMalloc(&d_leaves, sizeof(F)*LEAVE_WIDTH*n_digests_cap);
    cudaMalloc(&d_digests, sizeof(F)*HASH_WIDTH*n_digests_cap);
    cudaMemcpy(d_leaves, leaves, sizeof(F)*LEAVE_WIDTH*n_leaves, cudaMemcpyHostToDevice);

    device_fill_digests0<<<N_BLOCK, N_THREAD>>>(d_digests, d_leaves, n_leaves);
    cudaDeviceSynchronize();

    int level = 1;
    int n_level_leaves = n_leaves >> 1;
    uint32_t last_level_start_idx = 0;
    uint32_t level_start_idx = 2;

    while (n_level_leaves > (1 << cap_height)) {
        device_fill_digests1<<<N_BLOCK, N_THREAD>>>(d_digests, level, n_level_leaves, last_level_start_idx, level_start_idx);
        cudaDeviceSynchronize();

        level += 1;
        n_level_leaves = n_level_leaves >> 1;
        last_level_start_idx = level_start_idx;
        level_start_idx += (1<<level);
    }

    cudaMemcpy(digests_cap, d_digests, sizeof(F)*HASH_WIDTH*n_digests_cap, cudaMemcpyDeviceToHost);

    cudaFree(d_leaves);
    cudaFree(d_digests);

    // caps
    uint32_t left = last_level_start_idx;
    uint32_t right = left + 1;
    uint32_t to = level_start_idx;
    two_to_one(digests_cap, left, right, to);

    return;
}

void host_fill_digests_cap(F* digests_cap, F* leaves, uint32_t n_leaves, uint32_t cap_height) {
    F state[SPONGE_WIDTH] = { F(0) };

    for (uint32_t i=0; i<n_leaves; i++) {
        uint32_t from = i;
        uint32_t to = (i>>1<<2) | (i&0b1);
        permute(digests_cap, leaves, from, to);
    }

    uint32_t level = 1;
    uint32_t n_level_leaves = n_leaves >> 1;
    uint32_t last_level_start_idx = 0;
    uint32_t level_start_idx = 2;

    while (n_level_leaves > (1 << cap_height)) {
        for (uint32_t i=0; i<n_level_leaves; i++) {
            uint32_t left = last_level_start_idx + i*(1<<(level+1));
            uint32_t right = left + 1;
            uint32_t to = (level_start_idx + (i>>1)*(1<<(level+2))) | (i&0b1);
            two_to_one(digests_cap, left, right, to);
        }

        level += 1;
        n_level_leaves = n_level_leaves >> 1;
        last_level_start_idx = level_start_idx;
        level_start_idx += (1<<level);
    }

    // caps
    uint32_t left = last_level_start_idx;
    uint32_t right = left + 1;
    uint32_t to = level_start_idx;
    two_to_one(digests_cap, left, right, to);

    return;
}

void print_leaves(F* leaves, uint32_t n_leaves) {
    for (int i=0; i<n_leaves; i++) {
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

void print_digests(F* digests, uint32_t n_digests) {
    for (int i=0; i<n_digests; i++) {
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

void print_cap(F* digests_cap, uint32_t n_digests, uint32_t cap_height) {
    std::cout << std::hex;
    for (int i=0; i<(1<<cap_height); i++) {
        for (int j=0; j<HASH_WIDTH; j++) {
            std::cout << digests_cap[(n_digests+i)*HASH_WIDTH + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::dec;
}
