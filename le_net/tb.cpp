#include <cstdint>
#include <iostream>
#include <string>
#include <algorithm>
#include "le_net.h"

static data_t test_input[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH];
static data_t output_hw[OUTPUT_SIZE]; // Output from HLS function
static float test_expected_label;

std::string base_path = "C:/Users/Saksham/Documents/classwork/ECE554/LeNet-FPGA-HLS/data/MNIST/samples/";

void load_mnist_image(int test_num) {
    std::string filename = base_path + "image_" + std::to_string(test_num) + ".bin";
    FILE* file = fopen(filename.c_str(), "rb");
    if (file == nullptr) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    // read comma separated values into the input array
    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        for (int h = 0; h < INPUT_HEIGHT; ++h) {
            for (int w = 0; w < INPUT_WIDTH; ++w) {
                float value;
                fscanf(file, "%f,", &value);
                test_input[c][h][w] = (data_t)value;
            }
        }
    }
    fclose(file);

}

void load_mnist_label(int test_num) {
    std::string filename = base_path + "/label_" + std::to_string(test_num) + ".bin";
    FILE* file = fopen(filename.c_str(), "rb");
    if (file == nullptr) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    int value;
    fscanf(file, "%d", &value);
    test_expected_label = value;
    fclose(file);
}

bool run_test(int test_num) {
    load_mnist_image(test_num);
    load_mnist_label(test_num);

    lenet_hls(test_input, output_hw);

    int predicted_label = std::max_element(output_hw, output_hw + OUTPUT_SIZE) - output_hw;

    bool passed = predicted_label == test_expected_label;

    std::string pas = passed ? " Passed! " : " Failed! ";

    std::cout << test_num << pas << "- Predicted: " << predicted_label << ", Expected: " << test_expected_label << std::endl;

    return passed;
}


int main() {
    std::cout << "--- Starting LeNet HLS C-Simulation ---------------------------------" << std::endl;

    int count_passed = 0;
    int total_tests = 95;

    for (int i = 0; i < total_tests; ++i) {
        bool result = run_test(i);
        if (result) {
            count_passed++;
        }
    }

    std::cout << count_passed << "/" << total_tests << "(" << (100*(float)count_passed/(float)total_tests) << "%)" << " tests passed." << std::endl;
}
