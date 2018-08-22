#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

enum Feature
{
    SSE3,
    SSSE3,
    SSE4_1,
    SSE4_2,
    XSAVE,
    OSXSAVE,
    AVX,
    FPU,
    SSE,
    SSE2,
    SSE4A,
    AVX2,
};

enum Register { EAX, EBX, ECX, EDX };

struct FeatureInfo
{
	  // Input when calling cpuid (eax and ecx)
    const uint32_t func;
    const uint32_t subfunc;

    // Register and bit that holds the result
    const enum Register register_;
    const uint32_t bitmask;
};

struct FeatureInfo get_feature_info(enum Feature f)
{
    switch (f)
    {
      // ----------------------------------------------------------------------
      // EAX=1: Processor Info and Feature Bits
      // ----------------------------------------------------------------------
      case SSE3:         return (struct FeatureInfo) { 1, 0, ECX, 1u <<  0 }; // Streaming SIMD Extensions 3
      case SSSE3:        return (struct FeatureInfo) { 1, 0, ECX, 1u <<  9 }; // Supplemental Streaming SIMD Extensions 3
      case SSE4_1:       return (struct FeatureInfo) { 1, 0, ECX, 1u << 19 }; // Streaming SIMD Extensions 4.1
      case SSE4_2:       return (struct FeatureInfo) { 1, 0, ECX, 1u << 20 }; // Streaming SIMD Extensions 4.2
      case XSAVE:        return (struct FeatureInfo) { 1, 0, ECX, 1u << 26 }; // XSAVE/XSTOR States
      case OSXSAVE:      return (struct FeatureInfo) { 1, 0, ECX, 1u << 27 }; // OS Enabled Extended State Management
      case AVX:          return (struct FeatureInfo) { 1, 0, ECX, 1u << 28 }; // AVX Instructions

      case FPU:          return (struct FeatureInfo) { 1, 0, EDX, 1u <<  0 }; // Floating-Point Unit On-Chip

      case SSE:          return (struct FeatureInfo) { 1, 0, EDX, 1u << 25 }; // Streaming SIMD Extensions
      case SSE2:         return (struct FeatureInfo) { 1, 0, EDX, 1u << 26 }; // Streaming SIMD Extensions 2

      // ----------------------------------------------------------------------
      // EAX=80000001h: Extended Processor Info and Feature Bits (not complete)
      // ----------------------------------------------------------------------
      case SSE4A:        return (struct FeatureInfo) { 0x80000001, 0, ECX, 1u <<  6 }; // SSE4a
      case AVX2:         return (struct FeatureInfo) { 7, 0, ECX, 1u <<  5 }; // AVX2

    }

}

// Holds results of call to cpuid
struct cpuid_t
{
    uint32_t EAX;
    uint32_t EBX;
    uint32_t ECX;
    uint32_t EDX;
}; //< cpuid_t

static struct cpuid_t cpuid(uint32_t func, uint32_t subfunc)
{
    // Call cpuid
    // EBX/RBX needs to be preserved depending on the memory model and use of PIC
    struct cpuid_t result;
    asm volatile ("cpuid"
      : "=a"(result.EAX), "=b"(result.EBX), "=c"(result.ECX), "=d"(result.EDX)
      : "a"(func), "c"(subfunc));

    return result;
}

bool has_feature(enum Feature f)
{
    const struct FeatureInfo feature_info = get_feature_info(f);
	  const struct cpuid_t cpuid_result = cpuid(feature_info.func, feature_info.subfunc);

	  switch (feature_info.register_)
	  {
		  case EAX: return (cpuid_result.EAX & feature_info.bitmask) != 0;
		  case EBX: return (cpuid_result.EBX & feature_info.bitmask) != 0;
		  case ECX: return (cpuid_result.ECX & feature_info.bitmask) != 0;
		  case EDX: return (cpuid_result.EDX & feature_info.bitmask) != 0;
	  }
}

int main(int argc, char const *argv[])
{
	  printf("AVX2: %s\n", has_feature(AVX2) ? "yes" : "no");
	  return 0;
}
