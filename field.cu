#include "field.cuh"

// Addition
__host__ __device__
F F::operator+(const F& other) const {
    uint64_t sum0 = value + other.value;
    uint64_t sum1 = sum0 + ((sum0 < value || sum0 < other.value) ? EPSILON : 0);
    sum1 += (sum1 < sum0 || sum1 < EPSILON) ? EPSILON : 0;
    return F(sum1);
}

// Multiplication
__host__ __device__
F F::operator*(const F& other) const {
    uint128_t x = mul128(value, other.value);
    uint64_t y = reduce128(x);
    return F(y);
}

// Comparison
bool F::operator==(const F& other) const {
    return value == other.value;
}
