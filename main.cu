#include <chrono>
#include <iostream>
#include <stdio.h>

#include "poseidon.cuh"

#define N 100000

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

    int i = threadIdx.x;
    for (int j=0; j<WIDTH; j++) {
        state[j] = states[i*WIDTH + j];
    }

    poseidon(state);

    for (int j=0; j<WIDTH; j++) {
        states[i*WIDTH + j] = state[j];
    }

    return;
}

void print_debug(F* states) {
    std::cout << std::hex;
    for (int i=0; i<N; i++) {
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
    // print_debug(states);

    /******
       Init
    ******/
    for (int i=0; i<N; i++) {
        for (int j=0; j<WIDTH; j++) {
            states[i*WIDTH + j] = F(0);
        }
    }
    // print_debug(states);

    /********
       Device
    ********/
    start = std::chrono::high_resolution_clock::now();

    F* d_states;
    cudaMalloc(&d_states, sizeof(F)*N*WIDTH);
    cudaMemcpy(d_states, states, sizeof(F)*N*WIDTH, cudaMemcpyHostToDevice);
    devicePoseidon<<<1, N>>>(d_states);
    F* returned_states = (F*)malloc(sizeof(F)*N*WIDTH);
    cudaMemcpy(returned_states, d_states, sizeof(F)*N*WIDTH, cudaMemcpyDeviceToHost);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Device time is " << duration.count() << std::endl;
    // print_debug(returned_states);

    cudaFree(d_states);

    return 0;
}
