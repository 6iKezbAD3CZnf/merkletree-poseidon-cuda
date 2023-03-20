#include <chrono>
#include <iostream>

#include "merkle_tree.cuh"

#define CAP_HEIGHT 0
#define LOG_N_LEAVES 3

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    uint32_t cap_height = CAP_HEIGHT;
    uint32_t n_leaves = 1 << LOG_N_LEAVES;
    uint32_t leave_len = 83;
    uint32_t n_digests_cap = 2 * n_leaves - (1 << cap_height);

    /******
       Init
    *******/
    F* leaves = (F*)malloc(sizeof(F)*leave_len*n_leaves);
    for (int i=0; i<n_leaves; i++) {
        for (int j=0; j<leave_len; j++) {
            leaves[i*leave_len + j] = F(0);
        }
    }
    F* digests_cap = (F*)malloc(sizeof(F)*HASH_WIDTH*n_digests_cap);

    /******
       Host
    *******/
    start = std::chrono::high_resolution_clock::now();
    host_fill_digests_cap(
            digests_cap,
            leaves,
            n_leaves,
            leave_len,
            cap_height
            );
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Host time is " << duration.count() << std::endl;
    // print_leaves(leaves, n_leaves);
    // print_digests(digests_cap, n_digests_cap - (1<<cap_height));
    print_cap(digests_cap, n_digests_cap - (1<<cap_height), cap_height);

    /********
       Device
    *********/
    start = std::chrono::high_resolution_clock::now();
    device_fill_digests_cap(
            digests_cap,
            n_digests_cap,
            leaves,
            n_leaves,
            leave_len,
            cap_height
            );
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Device time is " << duration.count() << std::endl;
    // print_leaves(leaves, n_leaves);
    // print_digests(digests_cap, n_digests_cap - (1<<cap_height));
    print_cap(digests_cap, n_digests_cap - (1<<cap_height), cap_height);

    return 0;
}
