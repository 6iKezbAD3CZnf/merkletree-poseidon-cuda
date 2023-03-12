#ifndef UINT128_T_CUH_
#define UINT128_T_CUH_

#include <cstdint>

#define EPSILON 0xffffffff

class uint128_t{
public:
    __host__ __device__
    uint128_t() : lo(0), hi(0) {}

    __host__ __device__
    uint128_t(uint64_t n) : lo(n), hi(0) {}

    __host__ __device__
    uint128_t(uint64_t n, bool hi) {
        lo = hi ? 0 : n;
        hi = hi ? n : 0;
    }

    __host__ __device__
    uint128_t(uint64_t n0, uint64_t n1): lo(n0), hi(n1) {}

    __host__ __device__
    uint128_t & operator=(const uint128_t & n) {
        lo = n.lo;
        hi = n.hi;
        return * this;
    }

    __host__ __device__
    bool operator<(const uint128_t & n) {
        return ((hi == n.hi) ? (lo < n.lo) : (hi < n.hi));
    }

    __host__ __device__
    static inline uint128_t mul128(uint64_t x, uint64_t y) {
        uint128_t res;
      #ifdef __CUDA_ARCH__
        res.lo = x * y;
        res.hi = __umul64hi(x, y);
      #elif __x86_64__
        asm( "mulq %3\n\t"
             : "=a" (res.lo), "=d" (res.hi)
             : "%0" (x), "rm" (y));
      #else
      # error Architecture not supported
      #endif
        return res;
    }

    __host__ __device__
    static inline uint128_t add128(uint128_t x, uint128_t y) {
      #ifdef __CUDA_ARCH__
        uint128_t res;
        asm(  "add.cc.u64    %0, %2, %4;\n\t"
              "addc.u64      %1, %3, %5;\n\t"
              : "=l" (res.lo), "=l" (res.hi)
              : "l" (x.lo), "l" (x.hi),
                "l" (y.lo), "l" (y.hi));
        return res;
      #elif __x86_64__
        asm(  "add    %q2, %q0\n\t"
              "adc    %q3, %q1\n\t"
              : "+r" (x.lo), "+r" (x.hi)
              : "r" (y.lo), "r" (y.hi)
              : "cc");
        return x;
      #else
      # error Architecture not supported
      #endif
    }

    __host__ __device__
    static inline uint64_t reduce128(uint128_t x) {
        uint64_t x_hi_hi = x.hi >> 32;
        uint64_t x_hi_lo = x.hi & EPSILON;

        uint64_t t0 = x.lo - x_hi_hi;

        if (x.lo < x_hi_hi) {
            t0 -= EPSILON;
        }
        uint64_t t1 = x_hi_lo * EPSILON;

        uint64_t res_wrapped = t0 + t1;
        uint64_t t2;
        if (res_wrapped < t0 || res_wrapped < t1) {
            t2 = res_wrapped + EPSILON;
        } else {
            t2 = res_wrapped;
        }

        return t2;
    }

// private:
    uint64_t lo, hi;
}; // class uint128_t

__host__ __device__
inline uint128_t mul128(uint64_t x, uint64_t y)
{
    return uint128_t::mul128(x, y);
}

__host__ __device__
inline uint128_t add128(uint128_t x, uint128_t y)
{
    return uint128_t::add128(x, y);
}

__host__ __device__
inline uint64_t reduce128(uint128_t x)
{
    return uint128_t::reduce128(x);
}

#endif // UINT128_T_CUH_
