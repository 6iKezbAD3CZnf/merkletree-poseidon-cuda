#include <chrono>
#include <iostream>

#include "merkle_tree.cuh"

#define CAP_HEIGHT 4
#define LEAVE_LEN 135
#define LOG_N_LEAVES 20

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    uint32_t cap_height = CAP_HEIGHT;
    uint32_t num_leaves = 1 << LOG_N_LEAVES;
    uint32_t leave_len = LEAVE_LEN;
    uint32_t num_digests = 2 * (num_leaves - (1 << cap_height));
    uint32_t num_digests_caps = num_digests + (1 << cap_height);

    /******
       Init
    *******/
    F* leaves = (F*)malloc(sizeof(F)*leave_len*num_leaves);
    for (int i=0; i<num_leaves; i++) {
        for (int j=0; j<leave_len; j++) {
            leaves[i*leave_len + j] = F(0);
        }
    }
    F* digests_caps = (F*)malloc(sizeof(F)*HASH_WIDTH*num_digests_caps);

    /******
       Host
    *******/
    // start = std::chrono::high_resolution_clock::now();
    // host_fill_digests_caps(
    //         digests_caps,
    //         num_digests,
    //         leaves,
    //         num_leaves,
    //         leave_len,
    //         cap_height
    //         );
    // end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Host time is " << duration.count() << std::endl;
    // // print_leaves(leaves, num_leaves, leave_len);
    // // print_digests(digests_caps, num_digests);
    // print_caps(digests_caps, num_digests, cap_height);

    /******
       Init
    *******/
    for (int i=0; i<num_digests_caps; i++) {
        for (int j=0; j<HASH_WIDTH; j++) {
            digests_caps[i*HASH_WIDTH + j] = F(0);
        }
    }

    /********
       Device
    *********/
    start = std::chrono::high_resolution_clock::now();
    device_fill_digests_caps(
            digests_caps,
            num_digests,
            leaves,
            num_leaves,
            leave_len,
            cap_height
            );
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Device time is " << duration.count() << std::endl;
    // print_leaves(leaves, num_leaves, leave_len);
    // print_digests(digests_caps, num_digests);
    print_caps(digests_caps, num_digests, cap_height);

    return 0;
}
