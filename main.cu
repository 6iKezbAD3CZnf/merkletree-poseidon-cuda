#include <cassert>
#include <vector>

#define SPONGE_WIDTH 12
#define GOLDILOCKS_PRIME 0xffffffff0000001

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

F* poseidon(F* input) {
    return input;
}

void check_test_vectors(std::vector<unsigned long long*> inputs, std::vector<unsigned long long*> expected_outputs) {
    assert(inputs.size() == expected_outputs.size());

    for (int i=0; i<inputs.size(); i++) {
        unsigned long long* input_ = inputs[i];
        unsigned long long* expected_output_ = expected_outputs[i];

        F input[SPONGE_WIDTH] = {F(0)};
        for (int i=0; i<SPONGE_WIDTH; i++) {
            input[i] = F(input_[i]);
        }

        F* output = poseidon(input);

        for (int j=0; j<SPONGE_WIDTH; j++) {
            F ex_output = F(expected_output_[j]);
            assert(output[j] == ex_output);
        }
    }
}

void test_vectors() {
    unsigned long long input0[SPONGE_WIDTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    unsigned long long expected_output0[SPONGE_WIDTH] = {
         0x3c18a9786cb0b359, 0xc4055e3364a246c3, 0x7953db0ab48808f4, 0xc71603f33a1144ca,
         0xd7709673896996dc, 0x46a84e87642f44ed, 0xd032648251ee0b3c, 0x1c687363b207df62,
         0xdf8565563e8045fe, 0x40f5b37ff4254dae, 0xd070f637b431067c, 0x1792b1c4342109d7
    };

    std::vector<unsigned long long*> test_inputs = {input0};
    std::vector<unsigned long long*> expected_outputs = {expected_output0};

    check_test_vectors(test_inputs, expected_outputs);
}

int main() {
    test_vectors();
    return 0;
}
