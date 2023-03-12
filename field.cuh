#ifndef FIELD_CUH_
#define FIELD_CUH_

#include <iostream>
#include <cstdint>

#include "uint128_t.cuh"

#define GOLDILOCKS_PRIME 0xffffffff00000001

class F {
public:
    __host__ __device__
    F() : value(0) {}

    __host__ __device__
    F(uint64_t v) : value(v % GOLDILOCKS_PRIME) {}

    __host__ __device__
    F operator+(const F& other) const;

    __host__ __device__
    F operator*(const F& other) const;

    // Comparison
    bool operator==(const F& other) const;

    // Stream insertion operator
    friend std::ostream& operator<<(std::ostream& os, const F& f) {
        os << f.value;
        return os;
    }

    __host__ __device__
    uint64_t to_u64() {
        return value;
    }

private:
    uint64_t value;
};

#endif // FIELD_CUH_
