#include <cassert>

#include "merkle_tree.cuh"

extern "C"
F* digests_cap(F* leaves, uint32_t n_leaves, uint32_t leave_len, uint32_t cap_height) {
    uint32_t n_digests_cap = 2 * n_leaves - (1 << cap_height);

    F* digests_cap = (F*)malloc(sizeof(F)*HASH_WIDTH*n_digests_cap);
    host_fill_digests_cap(
            digests_cap,
            leaves,
            n_leaves,
            leave_len,
            cap_height
            );
    // device_fill_digests_cap(
    //         digests_cap,
    //         n_digests_cap,
    //         leaves,
    //         n_leaves,
    //         leave_len,
    //         cap_height
    //         );

    return digests_cap;
}
