#include <cstdint>
#include <iostream>
#include <string>
#include "le_net.h"

static float test_input[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH];
static float output_hw[OUTPUT_SIZE]; // Output from HLS function
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
                test_input[c][h][w] = value;
            }
        }
    }
    fclose(file);

    std::cout << "Read file: " << test_input[0][1][1] << std::endl;

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

    std::cout << test_num << " - Output probabilities: ";
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        std::cout << std::to_string(output_hw[i])<< ", ";
    }
    std::cout << std::endl;

    int predicted_label = 0;
    float max_prob = output_hw[0];
    for (int i = 1; i < OUTPUT_SIZE; ++i) {
        if (output_hw[i] > max_prob) {
            max_prob = output_hw[i];
            predicted_label = i;
        }
    }

    std::cout << test_num << " - Predicted Label: " << predicted_label << std::endl;
    std::cout << test_num << " - Expected Label: " << test_expected_label << std::endl;

    return predicted_label == test_expected_label;
}


int main() {
    std::cout << "--- Starting LeNet HLS C-Simulation ---------------------------------" << std::endl;

    // Debug: Check if weights are loaded
    std::cout << "Sample c1_weights[0][0][0][0] = " << c1_weights[0][0][0][0] << std::endl;
    std::cout << "Sample f5_weights[0][0] = " << f5_weights[0][0] << std::endl;
    std::cout << "Sample output_weights[0][0] = " << output_weights[0][0] << std::endl;

    int count_passed = 0;
    int total_tests = 95;

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
