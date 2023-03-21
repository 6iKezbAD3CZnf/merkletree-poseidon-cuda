#include <cassert>

#include "merkle_tree.cuh"

extern "C"
F* calculate_digests_caps(F* leaves, uint32_t num_leaves, uint32_t leave_len, uint32_t cap_height) {
    uint32_t num_caps = 1 << cap_height;
    uint32_t num_digests = 2 * (num_leaves - num_caps);
    uint32_t num_digests_caps = num_digests + num_caps;

    assert(num_leaves > num_caps);

    F* digests_cap = (F*)malloc(sizeof(F)*HASH_WIDTH*num_digests_caps);
    // host_fill_digests_caps(
    //         digests_cap,
    //         num_digests,
    //         leaves,
    //         num_leaves,
    //         leave_len,
    //         cap_height
    //         );
    device_fill_digests_caps(
            digests_cap,
            num_digests,
            leaves,
            num_leaves,
            leave_len,
            cap_height
            );

    return digests_cap;
}
