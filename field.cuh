#ifndef FIELD_CUH_
#define FIELD_CUH_

#define GOLDILOCKS_PRIME 0xffffffff0000001
#define SPONGE_WIDTH 12

class F {
public:
    F() : p(GOLDILOCKS_PRIME), value(0) {}
    F(unsigned long long v) : p(GOLDILOCKS_PRIME), value(v % p) {}

    // Addition
    F operator+(const F& other) const {
            return F((value + other.value) % p);
    }

    // Subtraction
    // F operator-(const F& other) const {
    //         return F((value - other.value + p) % p);
    // }

    // Multiplication
    // F operator*(const F& other) const {
    //         return F((value * other.value) % p);
    // }

    // Division
    // F operator/(const F& other) const {
    //         unsigned long long inv = inverse_mod(other.value, p);
    //         return F((value * inv) % p);
    // }

    // Exponentiation
    // F pow(unsigned long long exp) const {
    //     unsigned long long result = 1;
    //     unsigned long long base = value;
    //     while (exp > 0) {
    //         if (exp % 2 == 1) {
    //             result = (result * base) % p;
    //         }
    //         base = (base * base) % p;
    //         exp /= 2;
    //     }
    //     return F(result);
    // }

    // Comparison
    bool operator==(const F& other) const {
        return value == other.value;
    }
    bool operator!=(const F& other) const {
        return value != other.value;
    }

private:
    unsigned long long p;
    unsigned long long value;

    // Computes the modular inverse of a modulo p using the extended Euclidean algorithm
    // int inverse_mod(int a, int p) const {
    //     int t = 0, new_t = 1;
    //     int r = p, new_r = a;
    //     while (new_r != 0) {
    //         int quotient = r / new_r;
    //         int tmp_t = new_t;
    //         new_t = t - quotient * new_t;
    //         t = tmp_t;
    //         int tmp_r = new_r;
    //         new_r = r - quotient * new_r;
    //         r = tmp_r;
    //     }
    //     if (t < 0) {
    //         t += p;
    //     }
    //     return t;
    // }
};

#endif // FIELD_CUH_
