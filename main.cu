#include "poseidon.cuh"

#define N 1000000

F states[N][WIDTH];

void hostPoseidon(F states[N][WIDTH]) {

    for (int i=0; i<N; i++) {
        poseidon(states[i]);
    }

    return;
}

__global__ void devicePoseidon(F states[N][WIDTH]) {

    int i = threadIdx.x;
    poseidon(states[i]);

    return;
}

int main() {

    for (int i=0; i<N; i++) {
        for (int j=0; j<WIDTH; j++) {
            states[i][j] = F(0);
        }
    }

    hostPoseidon(states);
    devicePoseidon<<<1, N>>>(states);

    return 0;
}
