#include "le_net.h"

#include "hls_math.h"

const auto relu(data_t input) {
	return hls::max(input, (data_t)0.0f);
}


static void conv_c1(const data_t input[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH]) {
	C1_Filter_Loop: for (int f = 0; f < C1_CHANNELS; ++f) {
		C1_Row_Loop: for (int r = 0; r < C1_HEIGHT; ++r) {
			C1_Col_Loop: for (int c = 0; c < C1_WIDTH; ++c) {
				#pragma HLS PIPELINE
				data_t sum = 0.0f;
				C1_Depth_Loop: for (int d = 0; d < INPUT_CHANNELS; ++d) {
					C1_KRow_Loop: for (int kr = 0; kr < C1_KERNEL_SIZE; ++kr) {
						C1_KCol_Loop: for (int kc = 0; kc < C1_KERNEL_SIZE; ++kc) {
							sum += input[d][r + kr][c + kc] * c1_weights[f][d][kr][kc];
						}
					}
				}
				
				c1_output[f][r][c] = relu(sum + c1_bias[f]);
			}
		}
	}
}

static void pool_s2() {
	S2_Filter_Loop: for (int f = 0; f < S2_CHANNELS; ++f) {
		S2_Row_Loop: for (int r = 0; r < S2_HEIGHT; ++r) {
			S2_Col_Loop: for (int c = 0; c < S2_WIDTH; ++c) {
				#pragma HLS PIPELINE
				data_t sum = 0.0f;
				S2_PRow_Loop: for (int pr = 0; pr < S2_POOL_SIZE; ++pr) {
					S2_PCol_Loop: for (int pc = 0; pc < S2_POOL_SIZE; ++pc) {
						sum += c1_output[f][r * S2_POOL_SIZE + pr][c * S2_POOL_SIZE + pc];
					}
				}
				s2_output[f][r][c] = sum / (S2_POOL_SIZE * S2_POOL_SIZE);
			}
		}
	}
}

static void conv_c3() {
	C3_Filter_Loop: for (int f = 0; f < C3_CHANNELS; ++f) {
		C3_Row_Loop: for (int r = 0; r < C3_HEIGHT; ++r) {
			C3_Col_Loop: for (int c = 0; c < C3_WIDTH; ++c) {
				#pragma HLS PIPELINE
				data_t sum = 0.0f;
				C3_Depth_Loop: for (int d = 0; d < S2_CHANNELS; ++d) {
					C3_KRow_Loop: for (int kr = 0; kr < C3_KERNEL_SIZE; ++kr) {
						C3_KCol_Loop: for (int kc = 0; kc < C3_KERNEL_SIZE; ++kc) {
							sum += s2_output[d][r + kr][c + kc] * c3_weights[f][d][kr][kc];
						}
					}
				}
				c3_output[f][r][c] = relu(sum + c3_bias[f]);
			}
		}
	}
}

static void pool_s4() {
	S4_Filter_Loop: for (int f = 0; f < S4_CHANNELS; ++f) {
		S4_Row_Loop: for (int r = 0; r < S4_HEIGHT; ++r) {
			S4_Col_Loop: for (int c = 0; c < S4_WIDTH; ++c) {
				#pragma HLS PIPELINE
				data_t sum = 0.0f;
				S4_PRow_Loop: for (int pr = 0; pr < S4_POOL_SIZE; ++pr) {
					S4_PCol_Loop: for (int pc = 0; pc < S4_POOL_SIZE; ++pc) {
						sum += c3_output[f][r * S4_POOL_SIZE + pr][c * S4_POOL_SIZE + pc];
					}
				}
				s4_output[f][r][c] = sum / (S4_POOL_SIZE * S4_POOL_SIZE);
			}
		}
	}
}

static void flatten() {
	int index = 0;
	Flatten_Depth_Loop: for (int d = 0; d < S4_CHANNELS; ++d) {
		Flatten_Row_Loop: for (int r = 0; r < S4_HEIGHT; ++r) {
			Flatten_Col_Loop: for (int c = 0; c < S4_WIDTH; ++c) {
				#pragma HLS PIPELINE
				flatten_output[index++] = s4_output[d][r][c];
			}
		}
	}
}

static void fc_f5() {
	F5_Output_Loop: for (int o = 0; o < F5_OUTPUTS; ++o) {
		#pragma HLS PIPELINE
		data_t sum = 0.0f;
		F5_Input_Loop: for (int i = 0; i < F5_INPUTS; ++i) {
			sum += flatten_output[i] * f5_weights[o][i];
		}
		f5_output[o] = relu(sum + f5_bias[o]);
	}
}

static void fc_f6() {
	F6_Output_Loop: for (int o = 0; o < F6_OUTPUTS; ++o) {
		#pragma HLS PIPELINE
		data_t sum = 0.0f;
		F6_Input_Loop: for (int i = 0; i < F6_INPUTS; ++i) {
			sum += f5_output[i] * f6_weights[o][i];
		}
		f6_output[o] = relu(sum + f6_bias[o]);
	}
}

static void fc_output(data_t output[OUTPUT_SIZE]) {
	Output_Loop: for (int o = 0; o < OUTPUT_SIZE; ++o) {
		#pragma HLS PIPELINE
		data_t sum = 0.0f;
		Output_Input_Loop: for (int i = 0; i < F6_OUTPUTS; ++i) {
			sum += f6_output[i] * output_weights[o][i];
		}
		output[o] = sum + output_bias[o];
	}
}

static void softmax(data_t output[OUTPUT_SIZE]) {
	data_t max_val = output[0];
	Softmax_Max_Loop: for (int i = 1; i < OUTPUT_SIZE; ++i) {
		if (output[i] > max_val) {
			max_val = output[i];
		}
	}

	data_t sum_exp = 0.0f;
	Softmax_Exp_Loop: for (int i = 0; i < OUTPUT_SIZE; ++i) {
		#pragma HLS PIPELINE
		// Use HLS-synthesizable exp
		output[i] = hls::exp((output[i] - max_val));
		sum_exp += output[i];
	}

	Softmax_Norm_Loop: for (int i = 0; i < OUTPUT_SIZE; ++i) {
		#pragma HLS PIPELINE
		output[i] /= sum_exp;
	}
}

// --- Top-Level HLS Function ---
void lenet_hls(
	const data_t input[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH],
	data_t output[OUTPUT_SIZE]
) {

	// --- Execute the LeNet layers ---
	conv_c1(input);
	pool_s2();
	conv_c3();
	pool_s4();
	flatten();
	fc_f5();
	fc_f6();
	fc_output(output);	
	softmax(output);
}
