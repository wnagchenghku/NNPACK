#include <stdio.h>
#include <stdlib.h>

typedef struct {
              float* const T1;
              float* const T46;
        const float* const P2_saveMean;
        const float* const P2_saveVar;
        const float* const P2_runningMean;
        const float* const P2_runningVar;
        const float* const P2_weight;
        const float* const P2_bias;
} Tensors;

typedef struct {
        const int T4_1;
} Offsets;

void x86_FCA8A6B84(const int* const sizes, const Tensors* const tensors, const Offsets* const offsets) {
                /*x86_FCA8A6B84_00000A63(
                        0,
                        1,
                        sizes,
                        tensors->T1,
                        tensors->P2_bias,
                        tensors->P2_runningMean,
                        tensors->P2_runningVar,
                        tensors->P2_weight,
                        tensors->T46
                );*/
}



static void Function_CA8A6B84(float *T1, float *T46, float *P2_saveMean, float *P2_saveVar, float *P2_runningMean, float *P2_runningVar, float *P2_weight, float *P2_bias, int *sizes, int *offsets) {

        float *tensors[8] = {T1, T46, P2_saveMean, P2_saveVar, P2_runningMean, P2_runningVar, P2_weight, P2_bias};
        x86_FCA8A6B84(sizes, tensors, offsets);
        return;
}

void Forward_CA8A6B84(float *T1, float *T46, int *sizes, int *offsets) {
        float *P2_saveMean, *P2_saveVar;
        float P2_runningMean[64] = {0};
        float P2_runningVar[64] = {0};
        float P2_weight[64] = {0};
        float P2_bias[64] = {0};
        Function_CA8A6B84(T1, T46, P2_saveMean, P2_saveVar, P2_runningMean, P2_runningVar, P2_weight, P2_bias, sizes, offsets);
}

void ispc(void)
{
    float vin[16], vout[16];

    // Initialize input buffer
    int i;
    for (i = 0; i < 16; ++i)
        vin[i] = (int)i;

    // Call simple() function from simple.ispc file
    simple(vin, vout, 16);

    // Print results
    for (i = 0; i < 16; ++i)
        printf("%d: simple(%f) = %f\n", i, vin[i], vout[i]);
}

int main() {
    int sizes[4] = {1,3,224,224};
    int *offsets;
    offsets = (int[]){0};
    float *T1 = malloc(802816 * sizeof(float));
    float *T46 = malloc(sizeof(float) * 1 * 256 * 56 *56);
    Forward_CA8A6B84(T1, T46, sizes, offsets);
	ispc();
    sleep(2);
    printf("Hello World\n");
    return 0;
}
