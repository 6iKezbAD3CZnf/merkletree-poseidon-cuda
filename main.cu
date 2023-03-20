#include <chrono>
#include <iostream>

#include "merkle_tree.cuh"

#define CAP_HEIGHT 1
#define LEAVE_LEN 8
#define LOG_N_LEAVES 3

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    uint32_t cap_height = CAP_HEIGHT;
    uint32_t num_leaves = 1 << LOG_N_LEAVES;
    uint32_t leave_len = LEAVE_LEN;
    uint32_t num_digests = 2 * (num_leaves - (1 << cap_height));
    uint32_t num_digests_cap = num_digests + (1 << cap_height);

    /******
       Init
    *******/
    F* leaves = (F*)malloc(sizeof(F)*leave_len*num_leaves);
    for (int i=0; i<num_leaves; i++) {
        for (int j=0; j<leave_len; j++) {
            leaves[i*leave_len + j] = F(0);
        }
    }
    F* digests_cap = (F*)malloc(sizeof(F)*HASH_WIDTH*num_digests_cap);

    /******
       Host
    *******/
    start = std::chrono::high_resolution_clock::now();
    host_fill_digests_caps(
            digests_cap,
            num_digests,
            leaves,
            num_leaves,
            leave_len,
            cap_height
            );
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Host time is " << duration.count() << std::endl;
    // print_leaves(leaves, num_leaves);
    print_digests(digests_cap, num_digests_cap - (1<<cap_height));
    print_cap(digests_cap, num_digests_cap - (1<<cap_height), cap_height);

    /********
       Device
    *********/
    // start = std::chrono::high_resolution_clock::now();
    // device_fill_digests_cap(
    //         digests_cap,
    //         num_digests_cap,
    //         leaves,
    //         num_leaves,
    //         leave_len,
    //         cap_height
    //         );
    // end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Device time is " << duration.count() << std::endl;
    // // print_leaves(leaves, num_leaves);
    // // print_digests(digests_cap, num_digests_cap - (1<<cap_height));
    // print_cap(digests_cap, num_digests_cap - (1<<cap_height), cap_height);

    return 0;
}
