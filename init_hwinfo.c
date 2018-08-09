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

struct isa_info {
	bool has_avx;
	bool has_fma3;
	bool has_avx2;
};

struct cache_info {
	uint32_t size;
	uint32_t associativity;
	uint32_t threads;
	bool inclusive;
};

struct cache_hierarchy_info {
	struct cache_info l1;
	struct cache_info l2;
	struct cache_info l3;
	struct cache_info l4;
};

struct cache_blocking_info {
	size_t l1;
	size_t l2;
	size_t l3;
	size_t l4;
};

struct hardware_info {
	bool initialized;
	bool supported;
	uint32_t simd_width;

	struct cache_hierarchy_info cache;
	struct cache_blocking_info blocking;

	struct isa_info isa;
};

struct hardware_info nnp_hwinfo = { };

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

		
	/*
	 * Detect cache
	 */
	if (max_base_info >= 4) {
		for (uint32_t cache_id = 0; ; cache_id++) {
			struct cpu_info cache_info;
			__cpuid_count(4, cache_id, cache_info.eax, cache_info.ebx, cache_info.ecx, cache_info.edx);
			/* eax[bits 0-4]: cache type (0 - no more caches, 1 - data, 2 - instruction, 3 - unified) */
			const uint32_t type = cache_info.eax & 0x1F;
			if (type == 0) {
				break;
			} else if ((type == 1) || (type == 3)) {
				/* eax[bits 5-7]: cache level (starts at 1) */
				const uint32_t level = (cache_info.eax >> 5) & 0x7;
				/* eax[bits 14-25]: number of IDs for logical processors sharing the cache - 1 */
				const uint32_t threads = ((cache_info.eax >> 14) & 0xFFF) + 1;
				/* eax[bits 26-31]: number of IDs for processor cores in the physical package - 1 */
				const uint32_t cores = (cache_info.eax >> 26) + 1;

				/* ebx[bits 0-11]: line size - 1 */
				const uint32_t line_size = (cache_info.ebx & 0xFFF) + 1;
				/* ebx[bits 12-21]: line_partitions - 1 */
				const uint32_t line_partitions = ((cache_info.ebx >> 12) & 0x3FF) + 1;
				/* ebx[bits 22-31]: associativity - 1 */
				const uint32_t associativity = (cache_info.ebx >> 22) + 1;
				/* ecx: number of sets - 1 */
				const uint32_t sets = cache_info.ecx + 1;
				/* edx[bit 1]: cache inclusiveness */
				const bool inclusive = !!(cache_info.edx & 0x2);

				const struct cache_info cache_info = {
					.size = sets * associativity * line_partitions * line_size,
					.associativity = associativity,
					.threads = threads,
					.inclusive = inclusive,
				};
				switch (level) {
					case 1:
						nnp_hwinfo.cache.l1 = cache_info;
						break;
					case 2:
						nnp_hwinfo.cache.l2 = cache_info;
						break;
					case 3:
						nnp_hwinfo.cache.l3 = cache_info;
						break;
					case 4:
						nnp_hwinfo.cache.l4 = cache_info;
						break;
				}
			}
		}
	}
}

void init_hwinfo(void) {
	init_x86_hwinfo();
}