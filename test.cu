#include <cassert>

#include "test.cuh"

void check_test_vectors(std::vector<std::array<unsigned long long, SPONGE_WIDTH>> inputs, std::vector<std::array<unsigned long long, SPONGE_WIDTH>> expected_outputs) {
    assert(inputs.size() == expected_outputs.size());

    for (int i=0; i<inputs.size(); i++) {
        std::array<unsigned long long, SPONGE_WIDTH> input_ = inputs[i];
        std::array<unsigned long long, SPONGE_WIDTH> expected_output_ = expected_outputs[i];

        Hash input;
        for (int i=0; i<SPONGE_WIDTH; i++) {
            input[i] = F(input_[i]);
        }

        Hash output = poseidon(input);

        for (int j=0; j<SPONGE_WIDTH; j++) {
            F ex_output = F(expected_output_[j]);
            assert(output[j] == ex_output);
        }
    }
}

void test_vectors() {
    std::array<unsigned long long, SPONGE_WIDTH> input0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::array<unsigned long long, SPONGE_WIDTH> expected_output0 = {
         0x3c18a9786cb0b359, 0xc4055e3364a246c3, 0x7953db0ab48808f4, 0xc71603f33a1144ca,
         0xd7709673896996dc, 0x46a84e87642f44ed, 0xd032648251ee0b3c, 0x1c687363b207df62,
         0xdf8565563e8045fe, 0x40f5b37ff4254dae, 0xd070f637b431067c, 0x1792b1c4342109d7
    };

    std::vector<std::array<unsigned long long, SPONGE_WIDTH>> test_inputs = {input0};
    std::vector<std::array<unsigned long long, SPONGE_WIDTH>> expected_outputs = {expected_output0};

    check_test_vectors(test_inputs, expected_outputs);
}
