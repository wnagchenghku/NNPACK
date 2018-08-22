#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <cpuid.h>
#ifndef bit_AVX2
	#define bit_AVX2 0x00000020
#endif

struct isa_info {
	bool has_avx;
	bool has_fma3;
	bool has_avx2;
};

struct hardware_info {
	struct isa_info isa;
};

struct hardware_info nnp_hwinfo = { };

static inline uint64_t xgetbv(uint32_t ext_ctrl_reg) {
		uint32_t lo, hi;
		asm(".byte 0x0F, 0x01, 0xD0" : "=a" (lo), "=d" (hi) : "c" (ext_ctrl_reg));
		return (((uint64_t) hi) << 32) | (uint64_t) lo;
}

struct cpu_info {
		uint32_t eax;
		uint32_t ebx;
		uint32_t ecx;
		uint32_t edx;
};

static void init_x86_hwinfo(void) {
	const uint32_t max_base_info = __get_cpuid_max(0, NULL);
	const uint32_t max_extended_info = __get_cpuid_max(0x80000000, NULL);

	if (max_base_info >= 1) {
		struct cpu_info basic_info;
		__cpuid(1, basic_info.eax, basic_info.ebx, basic_info.ecx, basic_info.edx);

		/* OSXSAVE: ecx[bit 27] in basic info */
		const bool osxsave = !!(basic_info.ecx & bit_OSXSAVE);
		/* Check that AVX[bit 2] and SSE[bit 1] registers are preserved by OS */
		const bool ymm_regs = (osxsave ? ((xgetbv(0) & 0b110ul) == 0b110ul) : false);

		struct cpu_info structured_info = { 0 };
		if (max_base_info >= 7) {
			__cpuid_count(7, 0, structured_info.eax, structured_info.ebx, structured_info.ecx, structured_info.edx);
		}

		if (ymm_regs) {
			/* AVX: ecx[bit 28] in basic info */
			nnp_hwinfo.isa.has_avx  = !!(basic_info.ecx & bit_AVX);
			/* FMA3: ecx[bit 12] in basic info */
			nnp_hwinfo.isa.has_fma3 = !!(basic_info.ecx & bit_FMA);
			/* AVX2: ebx[bit 5] in structured feature info */
			nnp_hwinfo.isa.has_avx2 = !!(structured_info.ebx & bit_AVX2);
		}
	}
}

int main(int argc, char const *argv[])
{
	init_x86_hwinfo();
	if (nnp_hwinfo.isa.has_avx) {
		printf("AVX\n");
	}
	
	return 0;
}

