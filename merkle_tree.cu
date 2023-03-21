#include <cassert>
#include <cmath>

#include "merkle_tree.cuh"

__host__ __device__
void two_to_one(F* digest, F* left, F* right) {
    F state[SPONGE_WIDTH] = { F(0) };

    for (int k=0; k<SPONGE_WIDTH; k++) {
        if (k < HASH_WIDTH) {
            // left
            state[k] = left[k];
        } else if (k < 2*HASH_WIDTH) {
            // right
            state[k] = right[k - HASH_WIDTH];
        } else {
            state[k] = F(0);
        }
    }

    poseidon(state);

    for (int k=0; k<HASH_WIDTH; k++) {
        digest[k] = state[k];
    }
}

__host__ __device__
void hash_or_noop(F* digest, F* leave, uint32_t leave_len) {
    if (leave_len <= HASH_WIDTH) {
        // noop
        for (uint32_t i=0; i<HASH_WIDTH; i++) {
            if (i < leave_len) {
                digest[i] = leave[i];
            } else {
                digest[i] = F(0);
            }
        }

        return;
    }

    // hash_no_pad()
    // hash_n_to_hash_no_pad()
    // hash_n_to_m_no_pad()
    F state[SPONGE_WIDTH] = { F(0) };

    uint32_t quo = leave_len / SPONGE_RATE;
    uint32_t rem = leave_len % SPONGE_RATE;
    for (uint32_t i=0; i<quo; i++) {
        for (uint32_t j=0; j<SPONGE_RATE; j++) {
            state[j] = leave[i*SPONGE_RATE + j];
        }
        poseidon(state);
    }
    if (rem) {
        for (uint32_t i=0; i<rem; i++) {
            state[i] = leave[quo*SPONGE_RATE + i];
        }
        poseidon(state);
    }

    for (uint32_t i=0; i<HASH_WIDTH; i++) {
        digest[i] = state[i];
    }
}

__global__
void device_fill_digests0(
        F* d_digests_caps,
        uint32_t num_subtree_digests,
        F* d_leaves,
        uint32_t num_subtree_leaves,
        uint32_t leave_len,
        uint32_t num_caps
        ) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (id < num_caps * num_subtree_leaves) {
        int j = id % num_subtree_leaves; // outer loop
        int i = (id - j) / num_subtree_leaves; // inner loop

        uint32_t from = j;
        uint32_t to = (j>>1<<2) | (j&0b1);
        hash_or_noop(
                d_digests_caps + (num_subtree_digests*i + to)*HASH_WIDTH,
                d_leaves + (num_subtree_leaves*i + from)*leave_len,
                leave_len
                );

        id += stride;
    }
}

__global__
void device_fill_digests1(
        F* d_digests_caps,
        uint32_t num_subtree_digests,
        uint32_t level,
        uint32_t num_level_subtree_digests,
        uint32_t last_level_start_idx,
        uint32_t level_start_idx,
        uint32_t num_caps
) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (id < num_caps * num_level_subtree_digests) {
        int j = id % num_level_subtree_digests; // outer loop
        int i = (id - j) / num_level_subtree_digests; // inner loop

        uint32_t left = last_level_start_idx + j*(1<<(level+1));
        uint32_t right = left + 1;
        uint32_t to = (level_start_idx + (j>>1)*(1<<(level+2))) | (j&0b1);
        two_to_one(
                d_digests_caps + (num_subtree_digests*i + to)*HASH_WIDTH,
                d_digests_caps + (num_subtree_digests*i + left)*HASH_WIDTH,
                d_digests_caps + (num_subtree_digests*i + right)*HASH_WIDTH
                );

        id += stride;
    }

    return;
}

void device_fill_digests_caps(
        F* digests_caps,
        uint32_t num_digests,
        F* leaves,
        uint32_t num_leaves,
        uint32_t leave_len,
        uint32_t cap_height
        ) {
    uint32_t num_caps = 1 << cap_height;

    F* d_digests_caps;
    F* d_leaves;
    cudaMalloc(&d_leaves, sizeof(F)*leave_len*num_leaves);
    cudaMalloc(&d_digests_caps, sizeof(F)*HASH_WIDTH*(num_digests + num_caps));
    cudaMemcpy(d_leaves, leaves, sizeof(F)*leave_len*num_leaves, cudaMemcpyHostToDevice);

    device_fill_digests0<<<N_BLOCK, N_THREAD>>>(
            d_digests_caps,
            num_digests / num_caps,
            d_leaves,
            num_leaves / num_caps,
            leave_len,
            num_caps
            );
    cudaDeviceSynchronize();

    int level = 1;
    uint32_t num_level_digests = num_leaves >> 1;
    uint32_t last_level_start_idx = 0;
    uint32_t level_start_idx = 2;
    while (num_level_digests > num_caps) {
        device_fill_digests1<<<N_BLOCK, N_THREAD>>>(
                d_digests_caps,
                num_digests / num_caps,
                level,
                num_level_digests / num_caps,
                last_level_start_idx,
                level_start_idx,
                num_caps
                );
        cudaDeviceSynchronize();

        level += 1;
        num_level_digests = num_level_digests >> 1;
        last_level_start_idx = level_start_idx;
        level_start_idx += (1<<level);
    }

    cudaMemcpy(digests_caps, d_digests_caps, sizeof(F)*HASH_WIDTH*(num_digests + num_caps), cudaMemcpyDeviceToHost);

    cudaFree(d_leaves);
    cudaFree(d_digests_caps);

    // caps
    for (uint32_t i=0; i<num_caps; i++) {
        uint32_t subtree_digests_idx = num_digests / num_caps * i;
        uint32_t left = last_level_start_idx;
        uint32_t right = left + 1;
        two_to_one(
                digests_caps + (num_digests + i)*HASH_WIDTH,
                digests_caps + (subtree_digests_idx + left)*HASH_WIDTH,
                digests_caps + (subtree_digests_idx + right)*HASH_WIDTH
                );
    }

    return;
}

void host_fill_digests_caps_sub(
        uint32_t subtree_leaves_idx,
        uint32_t subtree_digests_idx,
        uint32_t cap_idx,
        F* digests_caps,
        F* leaves,
        uint32_t num_leaves,
        uint32_t leave_len
        ) {
    for (uint32_t i=0; i<num_leaves; i++) {
        uint32_t from = i;
        uint32_t to = (i>>1<<2) | (i&0b1);
        hash_or_noop(
                digests_caps + (subtree_digests_idx + to)*HASH_WIDTH,
                leaves + (subtree_leaves_idx + from)*leave_len,
                leave_len
                );
    }

    uint32_t level = 1;
    uint32_t num_level_leaves = num_leaves >> 1;
    uint32_t last_level_start_idx = 0;
    uint32_t level_start_idx = 2;

    while (num_level_leaves > 1) {
        for (uint32_t i=0; i<num_level_leaves; i++) {
            uint32_t left = last_level_start_idx + i*(1<<(level+1));
            uint32_t right = left + 1;
            uint32_t to = (level_start_idx + (i>>1)*(1<<(level+2))) | (i&0b1);
            two_to_one(
                    digests_caps + (subtree_digests_idx + to)*HASH_WIDTH,
                    digests_caps + (subtree_digests_idx + left)*HASH_WIDTH,
                    digests_caps + (subtree_digests_idx + right)*HASH_WIDTH
                    );
        }

        level += 1;
        num_level_leaves = num_level_leaves >> 1;
        last_level_start_idx = level_start_idx;
        level_start_idx += (1<<level);
    }

    // caps
    uint32_t left = last_level_start_idx;
    uint32_t right = left + 1;
    two_to_one(
            digests_caps + cap_idx*HASH_WIDTH,
            digests_caps + (subtree_digests_idx + left)*HASH_WIDTH,
            digests_caps + (subtree_digests_idx + right)*HASH_WIDTH
            );

    return;
}

void host_fill_digests_caps(
        F* digests_caps,
        uint32_t num_digests,
        F* leaves,
        uint32_t num_leaves,
        uint32_t leave_len,
        uint32_t cap_height
) {
    uint32_t num_caps = 1 << cap_height;
    uint32_t num_subtree_leaves = num_leaves / num_caps;
    uint32_t num_subtree_digests = num_digests / num_caps;
    for (uint32_t i=0; i<num_caps; i++) {
        host_fill_digests_caps_sub(
            num_subtree_leaves * i,
            num_subtree_digests * i,
            num_digests + i,
            digests_caps,
            leaves,
            num_leaves / num_caps,
            leave_len
        );
    }

    return;
}

void print_leaves(F* leaves, uint32_t num_leaves, uint32_t leave_len) {
    for (uint32_t i=0; i<num_leaves; i++) {
        std::cout << std::dec;
        std::cout << "leave" << i << " is [";
        std::cout << std::hex;
        for (uint32_t j=0; j<leave_len; j++) {
            std::cout << leaves[i*leave_len + j] << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::dec;
}

void print_digests(F* digests, uint32_t num_digests) {
    for (uint32_t i=0; i<num_digests; i++) {
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

void print_caps(F* digests_caps, uint32_t num_digests, uint32_t cap_height) {
    std::cout << std::hex;
    for (int i=0; i<(1<<cap_height); i++) {
        for (int j=0; j<HASH_WIDTH; j++) {
            std::cout << digests_caps[(num_digests+i)*HASH_WIDTH + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::dec;
}
