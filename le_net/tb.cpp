#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include "le_net.h"

// --- Static test data ---
static fixed_t dummy_input[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH];
static fixed_t output_hw[OUTPUT_SIZE]; // Output from HLS function

void fill_dummy_input() {
    srand(static_cast<unsigned int>(time(0)));
    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        for (int r = 0; r < INPUT_HEIGHT; ++r) {
            for (int w = 0; w < INPUT_WIDTH; ++w) {
                // Generate float then cast to fixed_t
                float val = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 0.1f;
                dummy_input[c][r][w] = (fixed_t)val;
            }
        }
    }
}

int main() {
    std::cout << "--- Starting LeNet HLS C-Simulation (8-bit fixed-point) ---" << std::endl;
    
    // Fill dummy input data
    fill_dummy_input();

    std::cout << "Running LeNet HLS function..." << std::endl;
    
    // --- Call the Hardware Function (Device Under Test) ---
    lenet_hls(dummy_input, output_hw);
    
    std::cout << "\nSoftmax Probabilities (from HLS function):" << std::endl;
    // ap_fixed types can be printed directly to iostream
    std::cout << std::fixed << std::setprecision(6);
    
    fixed_t sum_probs = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        std::cout << "Class " << i << ": " << output_hw[i] << std::endl;
        sum_probs += output_hw[i];
    }
    std::cout << "Sum of probabilities: " << sum_probs << std::endl;

    // HLS testbenches should return 0 on success
    std::cout << "--- HLS C-Simulation Finished ---" << std::endl;
    return 0;
}