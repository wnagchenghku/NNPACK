
#include <stdint.h>
#include <stdio.h>
#include "FCA8A6B84.h"

extern void x86_FCA8A6B84_00000A63_avx(
	const int coreIdx,
	const int numCores,
	const int* sizes,
	const Offsets* offsets, 
	const float* T1, 
	const float* P2_bias, 
	const float* P2_runningMean, 
	const float* P2_runningVar, 
	const float* P2_weight, 
	float* T46
);

extern void x86_FCA8A6B84_00000A63_sse(
        const int coreIdx,
        const int numCores,
        const int* sizes,
        const Offsets* offsets,
        const float* T1,
        const float* P2_bias,
        const float* P2_runningMean,
        const float* P2_runningVar,
        const float* P2_weight,
        float* T46
);



void x86_FCA8A6B84(const int* const sizes, const Tensors* const tensors, const Offsets* const offsets) {
		x86_FCA8A6B84_00000A63_sse(
			0,
			1,
			sizes,
			offsets, 
			tensors->T1, 
			tensors->P2_bias, 
			tensors->P2_runningMean, 
			tensors->P2_runningVar, 
			tensors->P2_weight, 
			tensors->T46
		);

                x86_FCA8A6B84_00000A63_avx(
                        0,
                        1,
                        sizes,
                        offsets,
                        tensors->T1,
                        tensors->P2_bias,
                        tensors->P2_runningMean,
                        tensors->P2_runningVar,
                        tensors->P2_weight,
                        tensors->T46
                );
}
