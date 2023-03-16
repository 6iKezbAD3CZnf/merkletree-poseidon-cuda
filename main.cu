#include <chrono>
#include <iostream>

#include "merkle_tree.cuh"

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    /******
       Init
    *******/
    F* leaves = (F*)malloc(sizeof(F)*LEAVE_WIDTH*N_LEAVES);
    for (int i=0; i<N_LEAVES; i++) {
        for (int j=0; j<LEAVE_WIDTH; j++) {
            leaves[i*LEAVE_WIDTH + j] = F(0);
        }
    }

    /******
       Host
    *******/
    start = std::chrono::high_resolution_clock::now();
    MerkleTree host_tree = MerkleTree(true, leaves, 0);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Host time is " << duration.count() << std::endl;
    host_tree.print_leaves();
    host_tree.print_digests();
    host_tree.print_root();

    /********
       Device
    *********/
    // start = std::chrono::high_resolution_clock::now();
    // MerkleTree device_tree = MerkleTree(false, leaves, 0);
    // end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Device time is " << duration.count() << std::endl;
    // // device_tree.print_leaves();
    // // device_tree.print_digests();
    // device_tree.print_root();

    return 0;
}
