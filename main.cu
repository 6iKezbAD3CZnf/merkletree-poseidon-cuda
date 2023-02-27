#include <cassert>
#include <chrono>
#include <iostream>
#include <stdio.h>

#include "poseidon.cuh"

#define N 500000
#define N_BLOCK 7
#define N_THREAD 640

void hostPoseidon(F* states) {
    F state[WIDTH];

    for (int i=0; i<N; i++) {
        for (int j=0; j<WIDTH; j++) {
            state[j] = states[i*WIDTH + j];
        }

        poseidon(state);

        for (int j=0; j<WIDTH; j++) {
            states[i*WIDTH + j] = state[j];
        }
    }

    return;
}

__global__ void devicePoseidon(F* states) {
    F state[WIDTH];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < N) {
        for (int j=0; j<WIDTH; j++) {
            state[j] = states[i*WIDTH + j];
        }

        poseidon(state);

        for (int j=0; j<WIDTH; j++) {
            states[i*WIDTH + j] = state[j];
        }

        i += stride;
    }

    return;
}

void print_debug(F* states) {
    std::cout << std::hex;
    for (int i=0; i<10; i++) {
        for (int j=0; j<WIDTH; j++) {
            std::cout << states[i*WIDTH + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::dec;
}

int main() {
    F* states = (F*)malloc(sizeof(F)*N*WIDTH);

    /******
       Init
    ******/
    for (int i=0; i<N; i++) {
        for (int j=0; j<WIDTH; j++) {
            states[i*WIDTH + j] = F(0);
        }
    }
    // print_debug(states);

    /******
       Host
    ******/
    auto start = std::chrono::high_resolution_clock::now();

    hostPoseidon(states);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Host time is " << duration.count() << std::endl;

    F* h_returned_states = (F*)malloc(sizeof(F)*N*WIDTH);
    for (int i=0; i<N; i++) {
        for (int j=0; j<WIDTH; j++) {
            h_returned_states[i*WIDTH + j] = states[i*WIDTH + j];
        }
    }
    // print_debug(h_returned_states);

    /******
       Init
    ******/
    for (int i=0; i<N; i++) {
        for (int j=0; j<WIDTH; j++) {
            states[i*WIDTH + j] = F(0);
        }
    }

    /********
       Device
    ********/
    start = std::chrono::high_resolution_clock::now();

    F* d_states;
    cudaMalloc(&d_states, sizeof(F)*N*WIDTH);
    cudaMemcpy(d_states, states, sizeof(F)*N*WIDTH, cudaMemcpyHostToDevice);
    devicePoseidon<<<N_BLOCK, N_THREAD>>>(d_states);
    F* d_returned_states = (F*)malloc(sizeof(F)*N*WIDTH);
    cudaMemcpy(d_returned_states, d_states, sizeof(F)*N*WIDTH, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Device time is " << duration.count() << std::endl;
    // print_debug(d_returned_states);

    /**************
       Sanity Check
    **************/
    for (int i=0; i<N; i++) {
        for (int j=0; j<WIDTH; j++) {
            // std::cout << i << std::endl;
            assert(h_returned_states[i*WIDTH + j] == d_returned_states[i*WIDTH + j]);
        }
    }

    cudaFree(d_states);
    free(h_returned_states);
    free(states);

    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    std::cout << "max_threads_per_block is " << max_threads_per_block << std::endl;

    return 0;
}
