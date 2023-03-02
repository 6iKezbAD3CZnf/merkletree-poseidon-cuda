#include <cassert>
#include <chrono>
#include <iostream>
#include <stdio.h>

#include "merkle_tree.cuh"

#define N_BLOCK 1
#define N_THREAD 1

// void hostPoseidon(F* states) {
//     F state[SPONGE_WIDTH];

//     for (int i=0; i<N; i++) {
//         for (int j=0; j<SPONGE_WIDTH; j++) {
//             state[j] = states[i*SPONGE_WIDTH + j];
//         }

//         poseidon(state);

//         for (int j=0; j<SPONGE_WIDTH; j++) {
//             states[i*SPONGE_WIDTH + j] = state[j];
//         }
//     }

//     return;
// }

// __global__
// void devicePoseidon(F* states) {
//     F state[SPONGE_WIDTH];

//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int stride = blockDim.x * gridDim.x;

//     while (i < N) {
//         for (int j=0; j<SPONGE_WIDTH; j++) {
//             state[j] = states[i*SPONGE_WIDTH + j];
//         }

//         poseidon(state);

//         for (int j=0; j<SPONGE_WIDTH; j++) {
//             states[i*SPONGE_WIDTH + j] = state[j];
//         }

//         i += stride;
//     }

//     return;
// }

void print_debug(F* leaves) {
    std::cout << std::hex;
    // for (int i=0; i<N; i++) {
    for (int i=0; i<1; i++) {
        for (int j=0; j<LEAVE_WIDTH; j++) {
            std::cout << leaves[i*LEAVE_WIDTH + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::dec;
}

int main() {
    F* leaves = (F*)malloc(sizeof(F)*LEAVE_WIDTH*N_LEAVES);

    /******
       Init
    ******/
    for (int i=0; i<N_LEAVES; i++) {
        for (int j=0; j<LEAVE_WIDTH; j++) {
            leaves[i*LEAVE_WIDTH + j] = F(0);
        }
    }

    MerkleTree tree = MerkleTree(leaves);
    tree.print_root();

    // [>*****
    //    Host
    // ******/
    // auto start = std::chrono::high_resolution_clock::now();

    // hostPoseidon(states);

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Host time is " << duration.count() << std::endl;

    // F* h_returned_states = (F*)malloc(sizeof(F)*N*SPONGE_WIDTH);
    // for (int i=0; i<N; i++) {
    //     for (int j=0; j<SPONGE_WIDTH; j++) {
    //         h_returned_states[i*SPONGE_WIDTH + j] = states[i*SPONGE_WIDTH + j];
    //     }
    // }
    // print_debug(h_returned_states);

    // [>*****
    //    Init
    // ******/
    // for (int i=0; i<N; i++) {
    //     for (int j=0; j<SPONGE_WIDTH; j++) {
    //         states[i*SPONGE_WIDTH + j] = F(0);
    //     }
    // }

    // [>*******
    //    Device
    // ********/
    // F* d_states;
    // cudaMalloc(&d_states, sizeof(F)*N*SPONGE_WIDTH);
    // F* d_returned_states = (F*)malloc(sizeof(F)*N*SPONGE_WIDTH);

    // start = std::chrono::high_resolution_clock::now();

    // cudaMemcpy(d_states, states, sizeof(F)*N*SPONGE_WIDTH, cudaMemcpyHostToDevice);
    // devicePoseidon<<<N_BLOCK, N_THREAD>>>(d_states);
    // cudaMemcpy(d_returned_states, d_states, sizeof(F)*N*SPONGE_WIDTH, cudaMemcpyDeviceToHost);

    // end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Device time is " << duration.count() << std::endl;
    // print_debug(d_returned_states);

    // [>*************
    //    Sanity Check
    // **************/
    // for (int i=0; i<N; i++) {
    //     for (int j=0; j<SPONGE_WIDTH; j++) {
    //         // std::cout << i << std::endl;
    //         assert(h_returned_states[i*SPONGE_WIDTH + j] == d_returned_states[i*SPONGE_WIDTH + j]);
    //     }
    // }

    // cudaFree(d_states);
    // free(h_returned_states);
    // free(states);

    return 0;
}
