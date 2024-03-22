#include "common.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	uint32_t N = 2 * ID.x;
	if (N + 1 + pushConstants.pass >= pushConstants.dataElementCount)
		return;

	float2 toSwap = vk::RawBufferLoad<float2>(pushConstants.bufferAddress + sizeof(float) * (N + pushConstants.pass));

	if (toSwap.x > toSwap.y) {
		toSwap = toSwap.yx;
	}

	vk::RawBufferStore<float32_t>(pushConstants.bufferAddress + sizeof(float) * (N + pushConstants.pass), toSwap);

}