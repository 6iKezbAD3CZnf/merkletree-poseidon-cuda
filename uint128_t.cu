#include "uint128_t.cuh"

__host__ __device__
uint128_t::uint128_t() : lo(0), hi(0) {}

__host__ __device__
uint128_t::uint128_t(uint64_t n) : lo(n), hi(0) {}

__host__ __device__
uint128_t & uint128_t::operator=(const uint128_t & n) {
    lo = n.lo;
    hi = n.hi;
    return * this;
}
