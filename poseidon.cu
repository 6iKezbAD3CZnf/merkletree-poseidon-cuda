#include <cassert>
#include <cstdint>
#include <iostream>

#include "poseidon.cuh"

void constant_layer(Hash& state, int round_ctr) {
    for (int i=0; i<SPONGE_WIDTH; i++) {
        unsigned long long round_constant = ALL_ROUND_CONSTANTS[i + SPONGE_WIDTH * round_ctr];
        state[i] = state[i] + F(round_constant);
        /* debug */ std::cout << "constant_layer." << i << " "; print(state);
    }
    return;
}

F sbox_monomial(F x) {
    // x |--> x^7
    F x2 = x.square();
    F x4 = x2.square();
    F x3 = x * x2;
    return x3 * x4;
}

void sbox_layer(Hash& state) {
    for (int i=0; i<SPONGE_WIDTH; i++) {
        state[i] = sbox_monomial(state[i]);
    }
}

F mds_row_shf(int r, Hash state) {
    assert(r < SPONGE_WIDTH);

    F res = F(0);

    for (int i=0; i<SPONGE_WIDTH; i++) {
        res = res + state[(i + r) % SPONGE_WIDTH] * MDS_MATRIX_CIRC[i];
    }
    res = res + state[r] * MDS_MATRIX_DIAG[r];

    return res;
}

Hash mds_layer(Hash& state) {
    Hash result;

    for (int r=0; r<SPONGE_WIDTH; r++) {
        result[r] = mds_row_shf(r, state);
    }

    return result;
}

void full_rounds(Hash& state, int *round_ctr) {
    for (int i=0; i<HALF_N_FULL_ROUNDS; i++) {
        constant_layer(state, *round_ctr);
        /* debug */ std::cout << "full_rounds." << i << " "; print(state);
        sbox_layer(state);
        state = mds_layer(state);
        *round_ctr += 1;
    }
    return;
}

// void partial_first_constant_layer(Hash& state) {
//     for (int i=0; i<SPONGE_WIDTH; i++) {
//         state[i] = state[i] + F(FAST_PARTIAL_FIRST_ROUND_CONSTANT[i]);
//     }
//     return;
// }

// Hash mds_partial_layer_init(Hash& state) {
//     Hash result;

//     return result;
// }

// Hash mds_partial_layer_fast(Hash& state, int i) {
//     Hash result;

//     return result;
// }

void partial_rounds(Hash& state, int *round_ctr) {
    // partial_first_constant_layer(state);
    // state = mds_partial_layer_init(state);

    // for (int i=0; i<N_PARTIAL_ROUNDS; i++) {
    //     state[0] = sbox_monomial(state[0]);
    //     state[0] = state[0] + F(FAST_PARTIAL_ROUND_CONSTANTS[i]);
    //     state = mds_partial_layer_fast(state, i);
    // }
    // *round_ctr += N_PARTIAL_ROUNDS;
    // return;
    for (int i=0; i<N_PARTIAL_ROUNDS; i++) {
        constant_layer(state, *round_ctr);
        state[0] = sbox_monomial(state[0]);
        state = mds_layer(state);
        *round_ctr += 1;
    }
    return;
}

// Poseidon Hash
Hash poseidon(Hash input) {
    Hash state = input;
    int round_ctr = 0;

    /* debug */ print(state);

    full_rounds(state, &round_ctr);

    /* debug */ print(state);

    partial_rounds(state, &round_ctr);
    full_rounds(state, &round_ctr);

    /* debug */ print(state);

    return state;
}

void print(Hash hash) {
    std::cout << "Hash is ";
    for (auto const& h: hash) {
        std::cout << std::hex << h << ", ";
    }
    std::cout << std::endl;
}
