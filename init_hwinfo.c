#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#if defined(__i386__) || defined(__x86_64__)
	#include <cpuid.h>
	#ifndef bit_AVX2
		#define bit_AVX2 0x00000020
	#endif
#endif

struct cpu_info {
	uint32_t eax;
	uint32_t ebx;
	uint32_t ecx;
	uint32_t edx;
};

static void init_x86_hwinfo(void) {
	const uint32_t max_base_info = __get_cpuid_max(0, NULL);
	const uint32_t max_extended_info = __get_cpuid_max(0x80000000, NULL);

	/*
	 * Detect CPU vendor
	 */
	struct cpu_info vendor_info;
	__cpuid(0, vendor_info.eax, vendor_info.ebx, vendor_info.ecx, vendor_info.edx);
	const uint32_t Auth = UINT32_C(0x68747541), enti = UINT32_C(0x69746E65), cAMD = UINT32_C(0x444D4163);
	const uint32_t Genu = UINT32_C(0x756E6547), ineI = UINT32_C(0x49656E69), ntel = UINT32_C(0x6C65746E);
	const uint32_t Cent = UINT32_C(0x746E6543), aurH = UINT32_C(0x48727561), auls = UINT32_C(0x736C7561);
	const bool is_intel = !((vendor_info.ebx ^ Genu) | (vendor_info.edx ^ ineI) | (vendor_info.ecx ^ ntel));
	const bool is_amd   = !((vendor_info.ebx ^ Auth) | (vendor_info.edx ^ enti) | (vendor_info.ecx ^ cAMD));
	const bool is_via   = !((vendor_info.ebx ^ Cent) | (vendor_info.edx ^ aurH) | (vendor_info.ecx ^ auls));
}

void init_hwinfo(void) {
	init_x86_hwinfo();
}