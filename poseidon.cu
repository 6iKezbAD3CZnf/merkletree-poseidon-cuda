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
        // printf("mds_row_shf: r=%d, i=%d, res=%lx\n", r, i, res.value);
        // printf("state[(i + r) % WIDTH] is %lx\n", state[(i+r)%WIDTH].value);
        // printf("MDS_MATRIX_CIRC[i] is %lx\n", MDS_MATRIX_CIRC[i]);
        F tmp = state[(i + r) % WIDTH] * MDS_MATRIX_CIRC[i];
        // printf("tmp is %lx\n", tmp.value);
        res = res + tmp;
    }
    res = res + state[r] * MDS_MATRIX_DIAG[r];

    // printf("mds_row_shf: r=%d, res=%lx\n", r, res.value);

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
    // for (int i=0; i<2; i++) {

        // printf("Initial ");
        // for (int j=0; j<WIDTH; j++) {
        //     printf("%lx, ", state[j].value);
        // }
        // printf("\n");

        constant_layer(state, *round_ctr);

        // printf("After constant ");
        // for (int j=0; j<WIDTH; j++) {
        //     printf("%lx, ", state[j].value);
        // }
        // printf("\n");

        sbox_layer(state);

        // printf("After sbox ");
        // for (int j=0; j<WIDTH; j++) {
        //     printf("%lx, ", state[j].value);
        // }
        // printf("\n");

        mds_layer(state);

        // printf("After mds ");
        // for (int j=0; j<WIDTH; j++) {
        //     printf("%lx, ", state[j].value);
        // }
        // printf("\n");

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

__host__ __device__
void partial_rounds(F state[WIDTH], int *round_ctr) {
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
        mds_layer(state);
        *round_ctr += 1;
    }
    return;
}

// Poseidon Hash
__host__ __device__
void poseidon(F state[WIDTH]) {
    int round_ctr = 0;

    printf("Initial ");
    for (int j=0; j<WIDTH; j++) {
        printf("%lx, ", state[j].value);
    }
    printf("\n");
    full_rounds(state, &round_ctr);
    printf("After fullrounds ");
    for (int j=0; j<WIDTH; j++) {
        printf("%lx, ", state[j].value);
    }
    printf("\n");
    partial_rounds(state, &round_ctr);
    printf("After partialrounds ");
    for (int j=0; j<WIDTH; j++) {
        printf("%lx, ", state[j].value);
    }
    printf("\n");
    full_rounds(state, &round_ctr);
    printf("After full roungds ");
    for (int j=0; j<WIDTH; j++) {
        printf("%lx, ", state[j].value);
    }
    printf("\n");
}
