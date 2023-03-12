#include "poseidon.cuh"

__host__ __device__
void constant_layer(F state[SPONGE_WIDTH], int round_ctr) {
    for (int i=0; i<SPONGE_WIDTH; i++) {
        uint64_t round_constant = ALL_ROUND_CONSTANTS[i + SPONGE_WIDTH * round_ctr];
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
void sbox_layer(F state[SPONGE_WIDTH]) {
    for (int i=0; i<SPONGE_WIDTH; i++) {
        state[i] = sbox_monomial(state[i]);
    }
}

__host__ __device__
F mds_row_shf(int r, F state[SPONGE_WIDTH]) {
    F res = F(0);

    for (int i=0; i<SPONGE_WIDTH; i++) {
        res = res + state[(i + r) % SPONGE_WIDTH] * MDS_MATRIX_CIRC[i];
    }
    res = res + state[r] * MDS_MATRIX_DIAG[r];

    return res;
}

__host__ __device__
void mds_layer(F state[SPONGE_WIDTH]) {
    F new_state[SPONGE_WIDTH];

    for (int r=0; r<SPONGE_WIDTH; r++) {
        new_state[r] = mds_row_shf(r, state);
    }

    for (int i=0; i<SPONGE_WIDTH; i++) {
        state[i] = new_state[i];
    }
}

__host__ __device__
void full_rounds(F state[SPONGE_WIDTH], int *round_ctr) {
    for (int i=0; i<HALF_N_FULL_ROUNDS; i++) {
        constant_layer(state, *round_ctr);
        sbox_layer(state);
        mds_layer(state);
        *round_ctr += 1;
    }
    return;
}

__host__ __device__
void partial_first_constant_layer(F state[SPONGE_WIDTH]) {
    for (int i=0; i<SPONGE_WIDTH; i++) {
        uint64_t constant = FAST_PARTIAL_FIRST_ROUND_CONSTANT[i];
        state[i] = state[i] + F(constant);
    }
    return;
}

__host__ __device__
void mds_partial_layer_init(F state[SPONGE_WIDTH]) {
    F new_state[SPONGE_WIDTH];

    new_state[0] = state[0];

    for (int r=1; r<SPONGE_WIDTH; r++) {
        for (int c=1; c<SPONGE_WIDTH; c++) {
            uint64_t constant = FAST_PARTIAL_ROUND_INITIAL_MATRIX[r-1][c-1];
            new_state[c] = new_state[c] + state[r] * F(constant);
        }
    }

    for (int i=0; i<SPONGE_WIDTH; i++) {
        state[i] = new_state[i];
    }
}

__host__ __device__
void add_u160_u128(uint128_t *x_lo, uint32_t *x_hi, uint128_t y) {
    uint128_t tmp_lo = add128(*x_lo, y);
    bool over = (tmp_lo < *x_lo) || (tmp_lo < y);
    *x_lo = tmp_lo;
    *x_hi = *x_hi + (over ? 1 : 0);
}

__host__ __device__
F reduce_u160(uint128_t n_lo, uint32_t n_hi) {
    uint64_t reduced_hi = reduce128(uint128_t(n_lo.hi, (uint64_t)n_hi));
    uint128_t reduced128 = uint128_t(n_lo.lo, reduced_hi);
    return F(reduce128(reduced128));
}

__host__ __device__
void mds_partial_layer_fast(F state[SPONGE_WIDTH], int r) {
    F new_state[SPONGE_WIDTH];

    uint128_t d_sum_lo = uint128_t((uint64_t) 0);
    uint32_t d_sum_hi = 0;

    for (int i=1; i<SPONGE_WIDTH; i++) {
        uint64_t t = FAST_PARTIAL_ROUND_W_HATS[r][i-1];
        uint64_t si = state[i].to_u64();
        add_u160_u128(&d_sum_lo, &d_sum_hi, mul128(si, t));
    }

    uint64_t s0 = state[0].to_u64();
    uint64_t mds0to0 = MDS_MATRIX_CIRC[0] + MDS_MATRIX_DIAG[0];
    add_u160_u128(&d_sum_lo, &d_sum_hi, mul128(s0, mds0to0));
    F d = reduce_u160(d_sum_lo, d_sum_hi);

    new_state[0] = d;
    for (int i=1; i<SPONGE_WIDTH; i++) {
        F t = F(FAST_PARTIAL_ROUND_VS[r][i-1]);
        new_state[i] = state[i] + state[0] * t;
    }

    for (int i=0; i<SPONGE_WIDTH; i++) {
        state[i] = new_state[i];
    }
}

__host__ __device__
void partial_rounds(F state[SPONGE_WIDTH], int *round_ctr) {
    // for (int i=0; i<N_PARTIAL_ROUNDS; i++) {
    //     constant_layer(state, *round_ctr);
    //     state[0] = sbox_monomial(state[0]);
    //     mds_layer(state);
    //     *round_ctr += 1;
    // }

    partial_first_constant_layer(state);
    mds_partial_layer_init(state);

    for (int i=0; i<N_PARTIAL_ROUNDS; i++) {
        state[0] = sbox_monomial(state[0]);
        state[0] = state[0] + F(FAST_PARTIAL_ROUND_CONSTANTS[i]);
        mds_partial_layer_fast(state, i);
    }
    *round_ctr += N_PARTIAL_ROUNDS;

    return;
}

// Poseidon Hash
__host__ __device__
void poseidon(F state[SPONGE_WIDTH]) {
    int round_ctr = 0;

    full_rounds(state, &round_ctr);
    partial_rounds(state, &round_ctr);
    full_rounds(state, &round_ctr);
}
