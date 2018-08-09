#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nnpack.h>

int main(int argc, char *argv[])
{
#if APPHELLOWORLD_PRINTARGS
	int i;

	printf("Arguments: ");
	for (i=0; i<argc; ++i)
		printf(" \"%s\"", argv[i]);
	printf("\n");
#endif
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
