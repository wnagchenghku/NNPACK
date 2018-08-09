#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nnpack.h>

#define bit_OSXSAVE (1 << 27)

#define __cpuid(level, a, b, c, d)			\
  __asm__ ("xchg{l}\t{%%}ebx, %1\n\t"			\
	   "cpuid\n\t"					\
	   "xchg{l}\t{%%}ebx, %1\n\t"			\
	   : "=a" (a), "=r" (b), "=c" (c), "=d" (d)	\
	   : "0" (level))

#define __cpuid_count(level, count, a, b, c, d)		\
  __asm__ ("xchg{l}\t{%%}ebx, %1\n\t"			\
	   "cpuid\n\t"					\
	   "xchg{l}\t{%%}ebx, %1\n\t"			\
	   : "=a" (a), "=r" (b), "=c" (c), "=d" (d)	\
	   : "0" (level), "2" (count))


static __inline unsigned int
__get_cpuid_max (unsigned int __ext, unsigned int *__sig)
{
  unsigned int __eax, __ebx, __ecx, __edx;

  /* Host supports cpuid.  Return highest supported cpuid input value.  */
  __cpuid (__ext, __eax, __ebx, __ecx, __edx);

  if (__sig)
    *__sig = __ebx;

  return __eax;
}

struct cpu_info_test {
	uint32_t eax;
	uint32_t ebx;
	uint32_t ecx;
	uint32_t edx;
};

static inline uint64_t xgetbv_test(uint32_t ext_ctrl_reg) {
			uint32_t lo, hi;
			asm(".byte 0x0F, 0x01, 0xD0" : "=a" (lo), "=d" (hi) : "c" (ext_ctrl_reg));
			return (((uint64_t) hi) << 32) | (uint64_t) lo;
}

void init_x86_hwinfo_test(){
	const uint32_t max_base_info = __get_cpuid_max(0, NULL);
	const uint32_t max_extended_info = __get_cpuid_max(0x80000000, NULL);

	printf("mini-os supports __get_cpuid_max\n");

	if (max_base_info >= 1) {
		struct cpu_info_test basic_info;
		__cpuid(1, basic_info.eax, basic_info.ebx, basic_info.ecx, basic_info.edx);

		printf("mini-os supports __cpuid\n");
		/* OSXSAVE: ecx[bit 27] in basic info */
		const bool osxsave = !!(basic_info.ecx & bit_OSXSAVE);
		/* Check that AVX[bit 2] and SSE[bit 1] registers are preserved by OS */
		const bool ymm_regs = (osxsave ? ((xgetbv_test(0) & 0b110ul) == 0b110ul) : false);

		printf("mini-os supports xgetbv\n");

		struct cpu_info_test structured_info = { 0 };
		if (max_base_info >= 7) {
			__cpuid_count(7, 0, structured_info.eax, structured_info.ebx, structured_info.ecx, structured_info.edx);
		}
		printf("mini-os supports __cpuid_count\n");
	}
}

int main(int argc, char *argv[])
{
	init_x86_hwinfo_test();
	printf("pass x86_hwinfo test\n");

	enum nnp_status init_status = nnp_initialize();
	if (init_status != nnp_status_success) {
		fprintf(stderr, "NNPACK initialization failed: error code %d\n", init_status);
		exit(EXIT_FAILURE);
	} else {
		fprintf(stderr, "NNPACK init true\n");
	}

	const size_t batch_size = 1;
	const size_t input_channels = 16;
	const size_t output_channels = 16;
	const struct nnp_padding input_padding = {0, 0, 0, 0};
	const struct nnp_size input_size = {180, 180};
	const struct nnp_size kernel_size = {3, 3};
	const struct nnp_size output_subsampling = {1, 1};
	const struct nnp_size output_size = {
		.width = (input_padding.left + input_size.width + input_padding.right - kernel_size.width) / output_subsampling.width + 1,
		.height = (input_padding.top + input_size.height + input_padding.bottom - kernel_size.height) / output_subsampling.height + 1
	};
	void* input = malloc(batch_size * input_channels * input_size.width * input_size.height * sizeof(float));
	void* kernel = malloc(input_channels * output_channels * kernel_size.width * kernel_size.height * sizeof(float));
	void* output = malloc(batch_size * output_channels * output_size.width * output_size.height * sizeof(float));
	void* bias = malloc(output_channels * sizeof(float));

	memset(input, 0, batch_size * input_channels * input_size.width * input_size.height * sizeof(float));
	memset(kernel, 0, input_channels * output_channels * kernel_size.width * kernel_size.height * sizeof(float));
	memset(output, 0, batch_size * output_channels * output_size.width * output_size.height * sizeof(float));
	memset(bias, 0, output_channels * sizeof(float));

	enum nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_auto;
	enum nnp_convolution_transform_strategy transform_strategy = nnp_convolution_transform_strategy_compute;
	enum nnp_status status = nnp_status_success;

	void *memory_block = NULL;
	size_t memory_size = 0;

	status = nnp_convolution_inference(
		algorithm, transform_strategy,
		input_channels, output_channels,
		input_size, input_padding, kernel_size, output_subsampling,
		NULL, NULL, NULL, NULL,
		NULL, &memory_size,
		nnp_activation_identity, NULL,
		NULL,
		NULL);

	if (memory_size != 0) {
		if (posix_memalign(&memory_block, 64, memory_size) !=0) {
			fprintf(stderr, "Error: failed to allocate %zu bytes for workspace\n", memory_size);
			exit(EXIT_FAILURE);
		}
		status = nnp_convolution_inference(
			algorithm, transform_strategy,
			input_channels, output_channels,
			input_size, input_padding, kernel_size, output_subsampling,
			input, kernel, bias, output,
			memory_block, &memory_size,
			nnp_activation_identity, NULL,
			NULL,
			NULL);
		free(memory_block);
	} else {
		status = nnp_convolution_inference(
			algorithm, transform_strategy,
			input_channels, output_channels,
			input_size, input_padding, kernel_size, output_subsampling,
			input, kernel, bias, output,
			NULL, NULL,
			nnp_activation_identity, NULL,
			NULL,
			NULL);	
	}

	if (status != nnp_status_success) {
		fprintf(stderr, "NNPACK nnp_convolution_inference failed: error code %d\n", status);
		exit(EXIT_FAILURE);
	}
	printf("Hello World\n");
}
