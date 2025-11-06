#pragma once

// Vitis HLS math library for synthesizable math functions
#include "hls_math.h"
// Vitis HLS library for fixed-point types
#include "ap_fixed.h"

// --- Define our 8-bit fixed-point type ---
// 8 total bits, 2 integer bits. Range is roughly [-2.0, 1.98]
typedef ap_fixed<8, 2> fixed_t;

// All preprocessor definitions remain the same.
#define INPUT_CHANNELS 1
#define INPUT_HEIGHT 32
#define INPUT_WIDTH 32

#define C1_KERNEL_SIZE 5
#define C1_CHANNELS 6
#define C1_HEIGHT (INPUT_HEIGHT - C1_KERNEL_SIZE + 1)
#define C1_WIDTH (INPUT_WIDTH - C1_KERNEL_SIZE + 1)

#define S2_POOL_SIZE 2
#define S2_CHANNELS C1_CHANNELS
#define S2_HEIGHT (C1_HEIGHT / S2_POOL_SIZE)
#define S2_WIDTH (C1_WIDTH / S2_POOL_SIZE)

#define C3_KERNEL_SIZE 5
#define C3_CHANNELS 16
#define C3_HEIGHT (S2_HEIGHT - C3_KERNEL_SIZE + 1)
#define C3_WIDTH (S2_WIDTH - C3_KERNEL_SIZE + 1)

#define S4_POOL_SIZE 2
#define S4_CHANNELS C3_CHANNELS
#define S4_HEIGHT (C3_HEIGHT / S4_POOL_SIZE)
#define S4_WIDTH (C3_WIDTH / S4_POOL_SIZE)

#define FLATTEN_SIZE (S4_CHANNELS * S4_HEIGHT * S4_WIDTH)

#define F5_INPUTS FLATTEN_SIZE
#define F5_OUTPUTS 120

#define F6_INPUTS F5_OUTPUTS
#define F6_OUTPUTS 84

#define OUTPUT_SIZE 10

// --- Top-Level HLS Function Prototype ---
// This is the function that will be synthesized into Verilog.
void lenet_hls(
    const fixed_t input[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH],
    fixed_t output[OUTPUT_SIZE]
);