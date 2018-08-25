
#include "FCA8A6B84.h"
#define FLT_MAX __FLT_MAX__

extern "C" unmasked void x86_FCA8A6B84_00000A63(
	const uniform int coreIdx,
	const uniform int numCores,
	const uniform int* uniform const sizes,
	const uniform Offsets* uniform const offsets, 
	const uniform float* uniform T1, 
	const uniform float* uniform P2_bias, 
	const uniform float* uniform P2_runningMean, 
	const uniform float* uniform P2_runningVar, 
	const uniform float* uniform P2_weight, 
	uniform float* uniform T46
) {
	for(uniform int core = coreIdx; core < 64; core += numCores) {
		const uniform int O1 = core % 64;
		uniform float T2[12544];
		for(int O2 = 0; O2 < 112; O2++) {
			foreach(O3 = 0 ... 112) {
				float T2_ = 0.0f;
				T2_ = ((T1[O1 * 12544 + O2 * 112 + O3] - P2_runningMean[O1]) * rsqrt((P2_runningVar[O1] + 1e-05f)));
				T2_ = ((T2_ * P2_weight[O1]) + P2_bias[O1]);
				T2[O2 * 112 + O3] = T2_;
			}
		}
		for(int O2 = 0; O2 < 56; O2++) {
			foreach(O3 = 0 ... 56) {
				float T46_ = 0.0f;
				for(int I2 = (O2 * 2) - 1; I2 < (O2 * 2) + 2; I2 += 1) {
					if(I2 >= 0 && I2 < 112) {
						for(int I3 = (O3 * 2) - 1; I3 < (O3 * 2) + 2; I3 += 1) {
							if(I3 >= 0 && I3 < 112) {
								T46_ = max(T46_, T2[I2 * 112 + I3]);
							}
						}
					}
				}
				T46[(offsets->T4_1 + O1) * 3136 + O2 * 56 + O3] = T46_;
			}
		}
	}
}
