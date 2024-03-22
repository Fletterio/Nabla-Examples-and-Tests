// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "../common/MonoDeviceApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"


#include "app_resources/common.hlsl"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

const uint32_t elementCount = 10000;

// this time instead of defining our own `int main()` we derive from `nbl::system::IApplicationFramework` to play "nice" wil all platforms
class BubbleSortApp final : public examples::MonoDeviceApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = examples::MonoDeviceApplication;
	using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;

	//We store a Compute Pipeline describing the tasks of the compute shader that does the BubbleSort work
	smart_refctd_ptr<IGPUComputePipeline> m_pipeline;

	//This utils class has some nice default upload/download buffers
	smart_refctd_ptr<nbl::video::IUtilities> m_utils;

	//We are going to have three buffers: One for upload, one for download, and one for the shader to work on as in/out
	smart_refctd_ptr<IGPUBuffer> m_deviceBuffer;
	StreamingTransientDataBufferMT<>* m_uploadBuffer;
	StreamingTransientDataBufferMT<>* m_downloadBuffer;

	// A `nbl::video::DeviceMemoryAllocator` is an interface to implement anything that can dish out free memory range to bind to back a `nbl::video::IGPUBuffer` or a `nbl::video::IGPUImage`
	// The Logical Device itself implements the interface and behaves as the most simple allocator, it will create a new `nbl::video::IDeviceMemoryAllocation` every single time.
	// We will cover allocators and suballocation in a later example.
	IDeviceMemoryAllocator::SAllocation m_deviceBufferAllocation = {};

	// Buffer Device Addresses for shader access
	uint64_t m_deviceBufferAddress;

	// You can ask the `nbl::core::GeneralpurposeAddressAllocator` used internally by the Streaming Buffers give out offsets aligned to a certain multiple (not only Power of Two!)
	uint32_t m_alignment;

	// Only Timeline Semaphores are supported in Nabla, there's no fences or binary semaphores.
	// Swapchains run on adaptors with empty submits that make them look like they work with Timeline Semaphores,
	// which has important side-effects we'll cover in another example.
	smart_refctd_ptr<ISemaphore> m_timeline;
	uint64_t m_iteration = 0;

public:

	BubbleSortApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	// we stuff all our work here because its a "single shot" app
	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		// this time we load a shader directly from a file
		smart_refctd_ptr<IGPUShader> shader;
		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = ""; // virtual root
			auto assetBundle = m_assetMgr->getAsset("app_resources/shader.comp.hlsl", lp);
			const auto assets = assetBundle.getContents();
			if (assets.empty())
				return logFail("Could not load shader!");

			// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
			auto source = IAsset::castDown<ICPUShader>(assets[0]);
			// The down-cast should not fail!
			assert(source);

			// this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
			shader = m_device->createShader(source.get());
			if (!shader)
				return logFail("Creation of a GPU Shader to from CPU Shader source failed!");
		}

		const nbl::asset::SPushConstantRange pcRange = { .stageFlags = IShader::ESS_COMPUTE,.offset = 0,.size = sizeof(PushConstantData) };

		// Nabla actually has facilities for SPIR-V Reflection and "guessing" pipeline layouts for a given SPIR-V which we'll cover in a different example
		smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pplnLayout = m_device->createPipelineLayout({ &pcRange, 1 });
		if (!pplnLayout)
			return logFail("Failed to create a Pipeline Layout!\n");
		{
			IGPUComputePipeline::SCreationParams params = {};
			params.layout = pplnLayout.get();
			// Theoretically a blob of SPIR-V can contain multiple named entry points and one has to be chosen, in practice most compilers only support outputting one (and glslang used to require it be called "main")
			params.shader.entryPoint = "main";
			params.shader.shader = shader.get();
			// we'll cover the specialization constant API in another example
			if (!m_device->createComputePipelines(nullptr, { &params,1 }, &m_pipeline))
				return logFail("Failed to create pipelines (compile & link shaders)!\n");
		}


		// Determine sizes of buffers created with the IUtilities class
		constexpr uint32_t BufferSize = sizeof(uint32_t) * elementCount;
		constexpr uint32_t DownloadBufferSize = BufferSize;
		constexpr uint32_t UploadBufferSize = BufferSize;

		// WARNING! Leaving upload/download buffer sizes by default to avoid having to check for a minimum size
		// If you try to modify this demo for too big a buffer you might need to add these back
		m_utils = make_smart_refctd_ptr<IUtilities>(smart_refctd_ptr(m_device), smart_refctd_ptr(m_logger));
		if (!m_utils)
			return logFail("Failed to create Utilities!");
		m_uploadBuffer = m_utils->getDefaultUpStreamingBuffer();
		m_downloadBuffer = m_utils->getDefaultDownStreamingBuffer();

		const auto& deviceLimits = m_device->getPhysicalDevice()->getLimits();
		// The ranges of non-coherent mapped memory you flush or invalidate need to be aligned. You'll often see a value of 64 reported by devices
		// which just happens to coincide with a CPU cache line size. So we ask our streaming buffers during allocation to give us properly aligned offsets.
		// Sidenote: For SSBOs, UBOs, BufferViews, Vertex Buffer Bindings, Acceleration Structure BDAs, Shader Binding Tables, Descriptor Buffers, etc.
		// there is also a requirement to bind buffers at offsets which have a certain alignment. Memory binding to Buffers and Images also has those.
		// We'll align to max of coherent atom size even if the memory is coherent,
		// and we also need to take into account BDA shader loads need to be aligned to the type being loaded.
		m_alignment = core::max(deviceLimits.nonCoherentAtomSize, alignof(float));

		// In contrast to fences, we just need one semaphore to rule all dispatches
		m_timeline = m_device->createSemaphore(m_iteration);

		// Get compute queue
		IQueue* const queue = getComputeQueue();

		// The allocators can do multiple allocations at once for efficiency
		const uint32_t AllocationCount = 1;

		// It comes with a certain drawback that you need to remember to initialize your "yet unallocated" offsets to the Invalid value
		// this is to allow a set of allocations to fail, and you to re-try after doing something to free up space without repacking args.
		auto inputOffset = m_uploadBuffer->invalid_value;

		// We always just wait till an allocation becomes possible (during allocation previous "latched" frees get their latch conditions polled)
		// Freeing of Streaming Buffer Allocations can and should be deferred until an associated polled event signals done (more on that later).
		std::chrono::steady_clock::time_point waitTill(std::chrono::years(45));
		// note that the API takes a time-point not a duration, because there are multiple waits and preemptions possible, so the durations wouldn't add up properly
		m_uploadBuffer->multi_allocate(waitTill, AllocationCount, &inputOffset, &UploadBufferSize, &m_alignment);

		{
			// Create a buffer with unique elements in [0, elementCount)
			uint32_t bufferData[elementCount];
			for (uint32_t i = 0; i < elementCount; i++) {
				bufferData[i] = i;
			}
			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(std::begin(bufferData), std::end(bufferData), g);

			// Get pointer to buffer start for CPU access
			auto* const inputPtr = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(m_uploadBuffer->getBufferPointer()) + inputOffset);

			// Upload buffer data to GPU
			memcpy(inputPtr, bufferData, sizeof(uint32_t) * elementCount);

			// Always remember to flush!
			if (m_uploadBuffer->needsManualFlushOrInvalidate())
			{
				const auto bound = m_uploadBuffer->getBuffer()->getBoundMemory();
				const ILogicalDevice::MappedMemoryRange range(bound.memory, bound.offset + inputOffset, BufferSize);
				m_device->flushMappedMemoryRanges(1, &range);
			}
		}

		// Allocate download buffer for CPU readback
		auto outputOffset = m_downloadBuffer->invalid_value;
		m_downloadBuffer->multi_allocate(waitTill, AllocationCount, &outputOffset, &DownloadBufferSize, &m_alignment);

		// Create the buffer the Compute Shader will use for the sort
		{
			// Always default the creation parameters, there's a lot of extra stuff for DirectX/CUDA interop and slotting into external engines you don't usually care about. 
			nbl::video::IGPUBuffer::SCreationParams params = {};
			params.size = BufferSize;
			// While the usages on `ICPUBuffers` are mere hints to our automated CPU-to-GPU conversion systems which need to be patched up anyway,
			// the usages on an `IGPUBuffer` are crucial to specify correctly.
			params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_SRC_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
			m_deviceBuffer = m_device->createBuffer(std::move(params));
			if (!m_deviceBuffer)
				return logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

			// Naming objects is cool because not only errors (such as Vulkan Validation Layers) will show their names, but RenderDoc captures too.
			m_deviceBuffer->setObjectDebugName("My Device Buffer");

			m_deviceBufferAddress = m_deviceBuffer->getDeviceAddress();

			// Get memory requirements for buffer memory creation
			nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = m_deviceBuffer->getMemoryReqs();

			// There are actually two `allocate` overloads, one which allocates memory if you already know the type you want.
			// And this one which is a utility which tries to allocate from every type that matches your requirements in some order of preference.
			// The other of preference (iteration over compatible types) can be controlled by the method's template parameter,
			// the default is from lowest index to highest, but skipping over incompatible types.
			m_deviceBufferAllocation = m_device->allocate(reqs, m_deviceBuffer.get(), nbl::video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
			if (!m_deviceBufferAllocation.isValid())
				return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

			// Note that we performed a Dedicated Allocation above, so there's no need to bind the memory anymore (since the allocator knows the dedication, it can already bind).
			// This is a carryover from having an OpenGL backend, where you couldn't have a memory allocation separate from the resource, so all allocations had to be "dedicated".
			// In Vulkan dedicated allocations are the most performant and still make sense as long as you won't blow the 4096 allocation limit on windows.
			// You should always use dedicated allocations for images used for swapchains, framebuffer attachments (esp transient), as well as objects used in CUDA/DirectX interop.
			assert(m_deviceBuffer->getBoundMemory().memory == m_deviceBufferAllocation.memory.get());
		}

		PushConstantData pcEven = PushConstantData{ m_deviceBufferAddress, elementCount, 0 };
		PushConstantData pcOdd = PushConstantData{ m_deviceBufferAddress, elementCount, 0 };

		// Create a one time submit command buffer to copy data from upload to the device buffer
		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> copyCmdBuf;
		// Similarly create a one time submit command buffer to copy data from device to download buffer
		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> copyBackCmdBuf;
		
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &copyCmdBuf))
				return logFail("Failed to create Command Buffers!\n");
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &copyBackCmdBuf))
				return logFail("Failed to create Command Buffers!\n");
		}

		// Copy data from upload to device buffer
		copyCmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		IGPUCommandBuffer::SBufferCopy copyInfo = {};
		copyInfo.size = BufferSize;
		copyInfo.srcOffset = inputOffset;
		copyCmdBuf->copyBuffer(m_uploadBuffer->getBuffer(), m_deviceBuffer.get(), 1, &copyInfo);
		copyCmdBuf->end();

		const IQueue::SSubmitInfo::SCommandBufferInfo copyCmdBufInfo =
		{
			.cmdbuf = copyCmdBuf.get()
		};

		const IQueue::SSubmitInfo::SSemaphoreInfo copySignalInfo =
		{
			.semaphore = m_timeline.get(),
			.value = m_iteration + 1,
			.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
		};

		const IQueue::SSubmitInfo copySubmitInfo = {
					.waitSemaphores = {},
					.commandBuffers = {&copyCmdBufInfo,1},
					.signalSemaphores = {&copySignalInfo,1}
		};

		queue->startCapture();
		queue->submit({ &copySubmitInfo, 1 });
		queue->endCapture();

		// Upload Buffer deallocation will be launched only after copy is done
		const ISemaphore::SWaitInfo copyWait = { m_timeline.get(), m_iteration + 1};

		// Deallocate upload buffer
		m_uploadBuffer->multi_deallocate(AllocationCount, &inputOffset, &UploadBufferSize, copyWait);

		//Advance timeline 
		m_iteration++;

		// Wait for copy to avoid setting up waits on the compute command buffer
		m_device->waitIdle();

		// Command Buffer where we submit all compute work
		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdBuf;

		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::NONE);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdBuf))
				return logFail("Failed to create Command Buffers!\n");
		}

		// Record the Command Buffer
		cmdBuf->begin(IGPUCommandBuffer::USAGE::NONE);
		// If you enable the `debugUtils` API Connection feature on a supported backend as we've done, you'll get these pretty debug sections in RenderDoc
		cmdBuf->beginDebugMarker("My Compute Dispatch", core::vectorSIMDf(0, 1, 0, 1));
		cmdBuf->bindComputePipeline(m_pipeline.get());
		// Even pass push constants
		cmdBuf->pushConstants(m_pipeline->getLayout(), IShader::ESS_COMPUTE, 0u, sizeof(pcEven), &pcEven);
		// Even pass dispatch
		cmdBuf->dispatch(ceil((float)elementCount / (WorkgroupSize * 2)), 1, 1);

		// We need to create a memory dependency between the even pass and the odd pass
		auto dependencyFlags = core::bitflag<asset::E_DEPENDENCY_FLAGS>(E_DEPENDENCY_FLAGS::EDF_NONE);
		IGPUCommandBuffer::SPipelineBarrierDependencyInfo dependencyInfo = {};
		const SMemoryBarrier memBarriers[] = { {.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, .srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
												.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, .dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS} };

		dependencyInfo.memBarriers = memBarriers;
		// Pipeline barrier for compute passes
		cmdBuf->pipelineBarrier(dependencyFlags, dependencyInfo);

		// Odd pass push constants
		cmdBuf->pushConstants(m_pipeline->getLayout(), IShader::ESS_COMPUTE, 0u, sizeof(pcOdd), &pcOdd);
		// Odd pass dispatch
		cmdBuf->dispatch(ceil((float)elementCount / (WorkgroupSize * 2)), 1, 1);

		cmdBuf->endDebugMarker();
		// Normally you'd want to perform a memory barrier when using the output of a compute shader or renderpass,
		// however signalling a timeline semaphore with the COMPUTE stage mask and waiting for it on the Host makes all Device writes visible.
		cmdBuf->end();

		m_logger->log("Semaphore is at: %d, timeline value is %d\n", ILogger::ELL_PERFORMANCE, m_timeline->getCounterValue(), m_iteration);

		for (auto i = 0u; i < elementCount; i++) {
			if (i % 100 == 0) {
				m_logger->log("On iteration %d", ILogger::ELL_PERFORMANCE, i);
			}

			IQueue::SSubmitInfo computeSubmitInfo[1] = {};

			const IQueue::SSubmitInfo::SCommandBufferInfo computeCmdBuf[] = { {.cmdbuf = cmdBuf.get()} };
			computeSubmitInfo[0].commandBuffers = computeCmdBuf;

			// We're going to signal the timeline semaphore and then have the device wait on it between full (even-odd) passes
			const IQueue::SSubmitInfo::SSemaphoreInfo computeSignals[] = { {.semaphore = m_timeline.get(),.value = m_iteration + 1,.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
			computeSubmitInfo[0].signalSemaphores = computeSignals;

			queue->startCapture();
			queue->submit(computeSubmitInfo);
			m_logger->log("Semaphore is at: %d, timeline value is %d\n", ILogger::ELL_PERFORMANCE, m_timeline->getCounterValue(), m_iteration);
			queue->endCapture();

			// As the name implies this function will not progress until the fence signals or repeated waiting returns an error.
			const ISemaphore::SWaitInfo waitInfos[] = { {
				.semaphore = m_timeline.get(),
				.value = m_iteration + 1
			} };
			m_device->blockForSemaphores(waitInfos);

			// Update iteration 
			m_iteration++;
		}

		// Copy data from device to download buffer
		copyBackCmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		IGPUCommandBuffer::SBufferCopy copyBackInfo = {};
		copyBackInfo.size = BufferSize;
		copyBackInfo.srcOffset = inputOffset;
		copyBackCmdBuf->copyBuffer(m_uploadBuffer->getBuffer(), m_deviceBuffer.get(), 1, &copyBackInfo);
		copyBackCmdBuf->end();

		IQueue::SSubmitInfo copyBackSubmitInfo[1] = {};
		const IQueue::SSubmitInfo::SCommandBufferInfo copyBackCmdBufInfo[] = { {.cmdbuf = copyBackCmdBuf.get()} };
		copyBackSubmitInfo[0].commandBuffers = copyBackCmdBufInfo;

		queue->startCapture();
		queue->submit(copyBackSubmitInfo);
		queue->endCapture();

		// Wait for copy
		m_device->waitIdle();

		// Create a one time submit command buffer for CPU readback of data
		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> readbackCmdBuf;
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &readbackCmdBuf))
				return logFail("Failed to create Command Buffers!\n");
		}
		// We let the readback latch know that the sorting is done
		const ISemaphore::SWaitInfo futureWait = { m_timeline.get(),m_iteration };

		// Stolen from the 05 example
		auto latchedConsumer = make_smart_refctd_ptr<IUtilities::CDownstreamingDataConsumer>(
			IDeviceMemoryAllocation::MemoryRange(outputOffset, BufferSize),
			// Note the use of capture by-value [=] and not by-reference [&] because this lambda will be called asynchronously whenever the event signals
			[=](const size_t dstOffset, const void* bufSrc, const size_t size)->void
			{
				// The unused variable is used for letting the consumer know the subsection of the output we've managed to download
				// But here we're sure we can get the whole thing in one go because we allocated the whole range ourselves.
				assert(dstOffset == 0 && size == BufferSize);

				// I can const cast, we know the mapping is just a pointer
				uint32_t* const data = reinterpret_cast<uint32_t*>(const_cast<void*>(bufSrc));
				std::string outBuffer;
				for (auto i = 0; i < elementCount / 20; i++) {
					for (auto j = 0; j < 20; j++) {
						auto index = 20 * i + j;
						if (index >= elementCount) continue;
						outBuffer.append(std::to_string(data[index]));
						outBuffer.append(" ");
					}
					outBuffer.append("\n");
				}
				m_logger->log("Your ordered array is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);
			},
			// Its also necessary to hold onto the commandbuffer because if it
			// hits its destructor, our automated reference counting will drop all references to objects used in the recorded commands.
			// It could also be latched in the upstreaming deallocate, because its the same fence.
			std::move(readbackCmdBuf), m_downloadBuffer
		);

		// We put a function we want to execute 
		m_downloadBuffer->multi_deallocate(AllocationCount, &outputOffset, &DownloadBufferSize, futureWait, &latchedConsumer.get());

		// There's just one caveat, the Queues tracking what resources get used in a submit do it via an event queue that needs to be polled to clear.
		// The tracking causes circular references from the resource back to the device, so unless we poll at the end of the application, they resources used by last submit will leak.
		// We could of-course make a very lazy thread that wakes up every second or so and runs this GC on the queues, but we think this is enough book-keeping for the users.
		//m_device->waitIdle();

		return true;
	}

	bool onAppTerminated() override
	{
		// Need to make sure that there are no events outstanding if we want all lambdas to eventually execute before `onAppTerminated`
		// (the destructors of the Streaming buffers will still wait for all lambda events to drain)
		while (m_downloadBuffer->cull_frees()) {}
		return device_base_t::onAppTerminated();
	}

	// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
	void workLoopBody() override {}

	// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
	bool keepRunning() override { return false; }

};


NBL_MAIN_FUNC(BubbleSortApp)