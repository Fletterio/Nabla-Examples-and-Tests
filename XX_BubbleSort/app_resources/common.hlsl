#include "nbl/builtin/hlsl/cpp_compat.hlsl"

struct PushConstantData
{
	uint64_t bufferAddress;
	uint32_t dataElementCount;
	uint32_t pass;
};

NBL_CONSTEXPR uint32_t WorkgroupSize = 256;