#pragma once

#include <cstdint>

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

static float c1_weights[C1_CHANNELS][INPUT_CHANNELS][C1_KERNEL_SIZE][C1_KERNEL_SIZE] = {
    #include "../c1_weights.bin"
};
static float c1_bias[C1_CHANNELS] = {
    #include "../c1_bias.bin"
};
static float c3_weights[C3_CHANNELS][S2_CHANNELS][C3_KERNEL_SIZE][C3_KERNEL_SIZE] = {
    #include "../c3_weights.bin"
};
static float c3_bias[C3_CHANNELS] = {
    #include "../c3_bias.bin"
};
static float f5_weights[F5_OUTPUTS][F5_INPUTS] = {
    #include "../f5_weights.bin"
};
static float f5_bias[F5_OUTPUTS] = {
    #include "../f5_bias.bin"
};

static float f6_weights[F6_OUTPUTS][F6_INPUTS] = {
    #include "../f6_weights.bin"
};
static float f6_bias[F6_OUTPUTS] = {
    #include "../f6_bias.bin"
};

static float output_weights[OUTPUT_SIZE][F6_OUTPUTS] = {
    #include "../output_weights.bin"
};
static float output_bias[OUTPUT_SIZE] = {
    #include "../output_bias.bin"
};

static float c1_output[C1_CHANNELS][C1_HEIGHT][C1_WIDTH];
static float s2_output[S2_CHANNELS][S2_HEIGHT][S2_WIDTH];
static float c3_output[C3_CHANNELS][C3_HEIGHT][C3_WIDTH];
static float s4_output[S4_CHANNELS][S4_HEIGHT][S4_WIDTH];
static float flatten_output[FLATTEN_SIZE];
static float f5_output[F5_OUTPUTS];
static float f6_output[F6_OUTPUTS];

// --- Top-Level HLS Function Prototype ---
// This is the function that will be synthesized into Verilog.
void lenet_hls(
    const float input[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH],
    float output[OUTPUT_SIZE]
);
