#ifndef FIELD_CUH_
#define FIELD_CUH_

#include <iostream>
#include <cstdint>

#include "uint128.cuh"

#define GOLDILOCKS_PRIME 0xffffffff00000001

class F {
public:
    __host__ __device__
    F() : p(GOLDILOCKS_PRIME), value(0) {}

    __host__ __device__
    F(uint64_t v) : p(GOLDILOCKS_PRIME), value(v % p) {}

    // Addition
    __host__ __device__
    F operator+(const F& other) const {
        uint64_t sum0 = value + other.value;
        uint64_t sum1 = sum0 + ((sum0 < value || sum0 < other.value) ? EPSILON : 0);
        sum1 += (sum1 < sum0 || sum1 < EPSILON) ? EPSILON : 0;
        return F(sum1);
    }

    // Multiplication
    __host__ __device__
    F operator*(const F& other) const {
        uint128_t x = mul128(value, other.value);
        uint64_t y = reduce128(x);
        return F(y);
    }

    // Comparison
    bool operator==(const F& other) const {
        return value == other.value;
    }
    // bool operator!=(const F& other) const {
    //     return value != other.value;
    // }

    // __host__ __device__
    // F square() const {
    //     uint128_t x = mul128(value, value);
    //     uint64_t y = reduce128(x);
    //     return F(y);
    // }

    // unsigned long long to_noncanonical_u64() const {
    //     return value;
    // }

    // Stream insertion operator
    friend std::ostream& operator<<(std::ostream& os, const F& f) {
        os << f.value;
        return os;
    }

// private:
    uint64_t p;
    uint64_t value;
};

#endif // FIELD_CUH_
