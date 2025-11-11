#include <cstdint>
#include <iostream>
#include <string>
#include "le_net.h"

static int8_t test_input[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH] = {0};
static int8_t output_hw[OUTPUT_SIZE] = {0}; // Output from HLS function
static int8_t test_expected_label;

std::string base_path = "C:/Users/Saksham/Documents/Classwork/ECE554/data/MNIST/samples/";

void load_mnist_image(int test_num) {
    std::string filename = base_path + "/image_" + std::to_string(test_num) + ".bin";
    FILE* file = fopen(filename.c_str(), "rb");
    if (file == nullptr) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    // read comma separated values into the input array
    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        for (int h = 0; h < INPUT_HEIGHT; ++h) {
            for (int w = 0; w < INPUT_WIDTH; ++w) {
                int value;
                fscanf(file, "%d,", &value);
                test_input[c][h][w] = static_cast<int8_t>(value);
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
    test_expected_label = static_cast<int8_t>(value);
    fclose(file);
}

bool run_test(int test_num) {
    load_mnist_image(test_num);
    load_mnist_label(test_num);

    lenet_hls(test_input, output_hw);

    std::cout << test_num << " - Output probabilities: ";
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        std::cout << std::to_string(output_hw[i])<< ", ";
    }
    std::cout << std::endl;

    int8_t predicted_label = -1;
    int8_t max_prob = INT8_MIN;
    for (int i = 1; i < OUTPUT_SIZE; ++i) {
        if (output_hw[i] > max_prob) {
            max_prob = output_hw[i];
            predicted_label = i;
        }
    }

    std::cout << test_num << " - Predicted Label: " << static_cast<int>(predicted_label) << std::endl;
    std::cout << test_num << " - Expected Label: " << static_cast<int>(test_expected_label) << std::endl;

    return predicted_label == test_expected_label;
}


int main() {
    std::cout << "--- Starting LeNet HLS C-Simulation ---------------------------------" << std::endl;

    int count_passed = 0;
    int total_tests = 10;

    for (int i = 0; i < total_tests; ++i) {
        bool result = run_test(i);
        if (result) {
            std::cout << "Test " << i << " Passed!" << std::endl;
            count_passed++;
        } else {
            std::cout << "Test " << i << " Failed!" << std::endl;
        }
    }

    std::cout << count_passed << "/" << total_tests << "(" << (100*(float)count_passed/(float)total_tests) << "%)" << " tests passed." << std::endl;


    std::cout << "--- HLS C-Simulation Finished ---------------------------------------" << std::endl;
}
