#include "poseidon.cuh"

__host__ __device__
void constant_layer(F state[WIDTH], int round_ctr) {
    for (int i=0; i<WIDTH; i++) {
        uint64_t round_constant = ALL_ROUND_CONSTANTS[i + WIDTH * round_ctr];
        state[i] = state[i] + F(round_constant);
    }
    return;
}

__host__ __device__
F sbox_monomial(F x) {
    // x |--> x^7
    F x2 = x * x;
    F x4 = x2 * x2;
    F x3 = x * x2;
    return x3 * x4;
}

__host__ __device__
void sbox_layer(F state[WIDTH]) {
    for (int i=0; i<WIDTH; i++) {
        state[i] = sbox_monomial(state[i]);
    }
}

__host__ __device__
F mds_row_shf(int r, F state[WIDTH]) {
    F res = F(0);

    for (int i=0; i<WIDTH; i++) {
        res = res + state[(i + r) % WIDTH] * MDS_MATRIX_CIRC[i];
    }
    res = res + state[r] * MDS_MATRIX_DIAG[r];

    return res;
}

__host__ __device__
void mds_layer(F state[WIDTH]) {
    F new_state[WIDTH];

    for (int r=0; r<WIDTH; r++) {
        new_state[r] = mds_row_shf(r, state);
    }

    for (int i=0; i<WIDTH; i++) {
        state[i] = new_state[i];
    }
}

__host__ __device__
void full_rounds(F state[WIDTH], int *round_ctr) {
    for (int i=0; i<HALF_N_FULL_ROUNDS; i++) {
        constant_layer(state, *round_ctr);
        sbox_layer(state);
        mds_layer(state);
        *round_ctr += 1;
    }
    return;
}

__host__ __device__
void partial_rounds(F state[WIDTH], int *round_ctr) {
    for (int i=0; i<N_PARTIAL_ROUNDS; i++) {
        constant_layer(state, *round_ctr);
        state[0] = sbox_monomial(state[0]);
        mds_layer(state);
        *round_ctr += 1;
    }
    return;
}

// Poseidon Hash
__host__ __device__
void poseidon(F state[WIDTH]) {
    int round_ctr = 0;

    full_rounds(state, &round_ctr);
    partial_rounds(state, &round_ctr);
    full_rounds(state, &round_ctr);
}
