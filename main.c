#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define STB_DS_IMPLEMENTATION
#include "stb/stb_ds.h"
#define VK_NO_PROTOTYPES
#define VOLK_IMPLEMENTATION
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3native.h>
#define FAST_OBJ_IMPLEMENTATION
#include "external/fast_obj/fast_obj.h"
#include "external/volk/volk.h"

#include "external/cglm/include/cglm/cglm.h"
#define u32 uint32_t
#define VK_CHECK(call) \
	do \
	{ \
		VkResult result_ = call; \
		assert(result_ == VK_SUCCESS); \
	} while (0)
#ifndef ARRAYSIZE
#define ARRAYSIZE(array) (sizeof(array) / sizeof((array)[0]))
#endif

// Define vertex structure
typedef struct Vertex
{
	vec3 pos;      // Position
	vec3 normal;   // Normal
	vec2 texcoord; // Texture coordinate
} Vertex;

typedef struct
{
	unsigned int id;
	const char* type;
} Texture;

typedef struct
{
	Vertex* vertices;
	unsigned int* indices;
	size_t num_v, num_i;
} Mesh;

typedef struct Buffer
{
	VkBuffer vkbuffer;
	VkDeviceMemory memory;
	void* data;
	size_t size;
} Buffer;

u32 selectmemorytype(
    VkPhysicalDeviceMemoryProperties* memprops, u32 memtypeBits, VkFlags requirements_mask)
{
	for (u32 i = 0; i < memprops->memoryTypeCount; ++i)
	{
		if ((memtypeBits & 1) == 1)
		{
			if ((memprops->memoryTypes[i].propertyFlags & requirements_mask) ==
			    requirements_mask)
			{
				return i;
			}
		}
		memtypeBits >>= 1;
	}
	assert(0 && "No suitable memory type found");
	return 0;
}

void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties* memprops, Buffer* buffer, size_t size, VkBufferUsageFlags usage)
{
	VkBufferCreateInfo bufferInfo = {
	    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	    .size = size,
	    .usage = usage,
	    .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
	VK_CHECK(vkCreateBuffer(device, &bufferInfo, NULL, &buffer->vkbuffer));
	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(device, buffer->vkbuffer, &memRequirements);
	VkMemoryAllocateInfo allocInfo = {
	    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	    .allocationSize = memRequirements.size,
	    .memoryTypeIndex = selectmemorytype(
	        memprops, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
	VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL, &buffer->memory));
	VK_CHECK(vkBindBufferMemory(device, buffer->vkbuffer, buffer->memory, 0));

	VK_CHECK(vkMapMemory(device, buffer->memory, 0, size, 0, &buffer->data));

	buffer->size = size;
}
void destroyBuffer(VkDevice device, Buffer* buffer)
{
	if (buffer->data)
	{
		vkUnmapMemory(device, buffer->memory);
		buffer->data = NULL;
	}
	if (buffer->vkbuffer)
	{
		vkDestroyBuffer(device, buffer->vkbuffer, NULL);
	}
	if (buffer->memory)
	{
		vkFreeMemory(device, buffer->memory, NULL);
	}
}

VkShaderModule LoadShaderModule(const char* filepath, VkDevice device)
{
	FILE* file = fopen(filepath, "rb");
	assert(file);

	fseek(file, 0, SEEK_END);
	long length = ftell(file);
	assert(length >= 0);
	fseek(file, 0, SEEK_SET);

	char* buffer = (char*)malloc(length);
	assert(buffer);

	size_t rc = fread(buffer, 1, length, file);
	assert(rc == (size_t)length);
	fclose(file);

	VkShaderModuleCreateInfo createInfo = {0};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = length;
	createInfo.pCode = (const uint32_t*)buffer;

	VkShaderModule shaderModule;
	VK_CHECK(vkCreateShaderModule(device, &createInfo, NULL, &shaderModule));

	free(buffer);
	return shaderModule;
}
// Add these near other Vulkan resource declarations
VkImage depthImage;              // NEW
VkDeviceMemory depthImageMemory; // NEW
VkImageView depthImageView;      // NEW
VkFormat depthFormat;            // NEW

// Add this function before main()
VkFormat findDepthFormat(VkPhysicalDevice physicalDevice)
{ // NEW
	const VkFormat candidates[] = {
	    VK_FORMAT_D32_SFLOAT,
	    VK_FORMAT_D32_SFLOAT_S8_UINT,
	    VK_FORMAT_D24_UNORM_S8_UINT};

	for (size_t i = 0; i < ARRAYSIZE(candidates); i++)
	{
		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, candidates[i], &props);

		if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			return candidates[i];
		}
	}

	assert(0 && "Failed to find supported depth format");
	return VK_FORMAT_UNDEFINED;
}

void createDepthResources(VkDevice device, VkPhysicalDevice physicalDevice, // NEW
    VkPhysicalDeviceMemoryProperties* memprops,
    uint32_t width, uint32_t height)
{
	depthFormat = findDepthFormat(physicalDevice);

	VkImageCreateInfo imageInfo = {
	    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
	    .imageType = VK_IMAGE_TYPE_2D,
	    .format = depthFormat,
	    .extent = {width, height, 1},
	    .mipLevels = 1,
	    .arrayLayers = 1,
	    .samples = VK_SAMPLE_COUNT_1_BIT,
	    .tiling = VK_IMAGE_TILING_OPTIMAL,
	    .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
	    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

	VK_CHECK(vkCreateImage(device, &imageInfo, NULL, &depthImage));

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(device, depthImage, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {
	    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	    .allocationSize = memRequirements.size,
	    .memoryTypeIndex = selectmemorytype(memprops, memRequirements.memoryTypeBits,
	        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};

	VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL, &depthImageMemory));
	VK_CHECK(vkBindImageMemory(device, depthImage, depthImageMemory, 0));

	VkImageViewCreateInfo viewInfo = {
	    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,

	    .image = depthImage,
	    .viewType = VK_IMAGE_VIEW_TYPE_2D,
	    .format = depthFormat,

	    .subresourceRange = {
	        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
	        .baseMipLevel = 0,
	        .levelCount = 1,
	        .baseArrayLayer = 0,
	        .layerCount = 1}};

	VK_CHECK(vkCreateImageView(device, &viewInfo, NULL, &depthImageView));
}
int main(int argc, const char** argv)
{

#if defined(VK_USE_PLATFORM_XLIB_KHR)
	glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
	glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
#endif
	int rc = glfwInit();
	assert(rc);
	VK_CHECK(volkInitialize());
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	GLFWwindow* window = glfwCreateWindow(800, 600, "ok", 0, 0);
	assert(window);
	int windowWidth = 0, windowHeight = 0;
	glfwGetWindowSize(window, &windowWidth, &windowHeight);
	VkApplicationInfo appInfo = {
	    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
	    .apiVersion = VK_API_VERSION_1_3,
	};
	VkInstanceCreateInfo createInfo = {
	    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
	    .pApplicationInfo = &appInfo,
	};
#ifdef _DEBUG
	const char* debugLayers[] = {"VK_LAYER_KHRONOS_validation"};
	createInfo.ppEnabledLayerNames = debugLayers;
	createInfo.enabledLayerCount = ARRAYSIZE(debugLayers);
#endif
	const char* extensions[] = {
	    VK_KHR_SURFACE_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
	    VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
	    VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
#endif
#ifndef NDEBUG
	    VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
#endif
	};
	createInfo.ppEnabledExtensionNames = extensions;
	createInfo.enabledExtensionCount = ARRAYSIZE(extensions);
	VkInstance instance;
	VK_CHECK(vkCreateInstance(&createInfo, 0, &instance));
	volkLoadInstance(instance);
	VkPhysicalDevice physicalDevices[8];
	u32 physicalDeviceCount = ARRAYSIZE(physicalDevices);
	VK_CHECK(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount,
	    physicalDevices));
	VkPhysicalDevice selectedPhysicalDevice = VK_NULL_HANDLE,
	                 discrete = VK_NULL_HANDLE, fallback = VK_NULL_HANDLE;
	for (u32 i = 0; i < physicalDeviceCount; ++i)
	{
		VkPhysicalDeviceProperties props = {0};
		vkGetPhysicalDeviceProperties(physicalDevices[i], &props);
		printf("GPU%d: %s\n", i, props.deviceName);
		discrete =
		    (!discrete && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		        ? physicalDevices[i]
		        : discrete;
		fallback = (!fallback) ? physicalDevices[i] : fallback;
	}
	selectedPhysicalDevice = discrete ? discrete : fallback;
	if (selectedPhysicalDevice)
	{
		VkPhysicalDeviceProperties props = {0};
		vkGetPhysicalDeviceProperties(selectedPhysicalDevice, &props);
		printf("Selected GPU: %s\n", props.deviceName);
	}
	else
	{
		printf("No suitable GPU found\n");
		exit(1);
	}
	u32 queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(selectedPhysicalDevice,
	    &queueFamilyCount, NULL);
	VkQueueFamilyProperties* queueFamilies =
	    malloc(queueFamilyCount * sizeof(VkQueueFamilyProperties));
	vkGetPhysicalDeviceQueueFamilyProperties(selectedPhysicalDevice,
	    &queueFamilyCount, queueFamilies);
	u32 queuefamilyIndex = UINT32_MAX;
	for (u32 i = 0; i < queueFamilyCount; ++i)
	{
		if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
		{
			queuefamilyIndex = i;
			break;
		}
	}
	assert(queuefamilyIndex != UINT32_MAX && "No suitable queue family found");
	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo queueCreateInfo = {
	    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
	    .queueFamilyIndex = queuefamilyIndex,
	    .queueCount = 1,
	    .pQueuePriorities = &queuePriority,
	};
	VkPhysicalDeviceFeatures deviceFeatures = {0};
	const char* deviceExtensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
	VkDeviceCreateInfo deviceCreateInfo = {
	    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
	    .queueCreateInfoCount = 1,
	    .pQueueCreateInfos = &queueCreateInfo,
	    .enabledExtensionCount = ARRAYSIZE(deviceExtensions),
	    .ppEnabledExtensionNames = deviceExtensions,
	    .pEnabledFeatures = &deviceFeatures,
	};

	VkPhysicalDeviceMemoryProperties memprops;
	vkGetPhysicalDeviceMemoryProperties(selectedPhysicalDevice, &memprops);

	VkDevice device;
	VK_CHECK(
	    vkCreateDevice(selectedPhysicalDevice, &deviceCreateInfo, 0, &device));
	volkLoadDevice(device);
	// surface createinfo need different for other os or x11
	VkSurfaceKHR surface = 0;
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
	VkWaylandSurfaceCreateInfoKHR surfacecreateInfo = {
	    .sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
	    .display = glfwGetWaylandDisplay(),
	    .surface = glfwGetWaylandWindow(window),
	};
	VK_CHECK(vkCreateWaylandSurfaceKHR(instance, &surfacecreateInfo, 0, &surface));
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
	VkXlibSurfaceCreateInfoKHR surfacecreateInfo = {
	    .sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
	    .dpy = glfwGetX11Display(),
	    .window = glfwGetX11Window(window),
	};
	VK_CHECK(vkCreateXlibSurfaceKHR(instance, &surfacecreateInfo, 0, &surface));
#else
	printf("No supported platform defined for Vulkan surface creation");
#endif

	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSetLayoutBinding uboLayoutBinding = {
	    .binding = 0,
	    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	    .descriptorCount = 1,
	    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT};

	VkDescriptorSetLayoutCreateInfo layoutInfo = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	    .bindingCount = 1,
	    .pBindings = &uboLayoutBinding};
	VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &descriptorSetLayout));
	// Add after device creation
	VkDescriptorPoolSize poolSize = {
	    .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	    .descriptorCount = 1};

	VkDescriptorPoolCreateInfo poolInfo = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
	    .maxSets = 1,
	    .poolSizeCount = 1,
	    .pPoolSizes = &poolSize};

	VkDescriptorPool descriptorPool;
	VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, NULL, &descriptorPool));
	// Add after descriptor pool creation
	VkDescriptorSetAllocateInfo allocInfo = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
	    .descriptorPool = descriptorPool,
	    .descriptorSetCount = 1,
	    .pSetLayouts = &descriptorSetLayout};

	VkDescriptorSet descriptorSet;
	VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

	VkBool32 presentSupported = 0;
	VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(
	    selectedPhysicalDevice, queuefamilyIndex, surface, &presentSupported));
	assert(presentSupported);

	VkSurfaceCapabilitiesKHR surfaceCapabilities;
	VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
	    selectedPhysicalDevice, surface, &surfaceCapabilities));
	u32 formatCount = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(selectedPhysicalDevice, surface,
	    &formatCount, NULL);
	VkSurfaceFormatKHR* formats =
	    malloc(formatCount * sizeof(VkSurfaceFormatKHR));
	vkGetPhysicalDeviceSurfaceFormatsKHR(selectedPhysicalDevice, surface,
	    &formatCount, formats);

	VkSwapchainKHR swapchain;
	VkSwapchainCreateInfoKHR swapchaincreateinfo = {
	    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
	    .surface = surface,
	    .minImageCount = surfaceCapabilities.minImageCount,
	    .imageFormat = formats[0].format,
	    .imageColorSpace = formats[0].colorSpace,
	    .imageExtent = {.width = windowWidth, .height = windowHeight},
	    .imageArrayLayers = 1,
	    .imageUsage =
	        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
	    .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
	    .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
	    .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
	    .presentMode = VK_PRESENT_MODE_FIFO_KHR,
	    .clipped = VK_TRUE,
	    .queueFamilyIndexCount = 1,
	    .pQueueFamilyIndices = &queuefamilyIndex,
	};
	VK_CHECK(vkCreateSwapchainKHR(device, &swapchaincreateinfo, 0, &swapchain));

	createDepthResources(device, selectedPhysicalDevice, &memprops, windowWidth, windowHeight); // NEW
	VkSemaphore imageAvailableSemaphore;
	VkSemaphore renderCompleteSemaphore;
	VkSemaphoreCreateInfo semInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
	VK_CHECK(vkCreateSemaphore(device, &semInfo, 0, &imageAvailableSemaphore));
	VK_CHECK(vkCreateSemaphore(device, &semInfo, 0, &renderCompleteSemaphore));
	VkQueue queue;
	vkGetDeviceQueue(device, queuefamilyIndex, 0, &queue);
	VkCommandPoolCreateInfo commandPoolInfo = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
	    .queueFamilyIndex = queuefamilyIndex,
	    .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
	};
	VkCommandPool commandpool;
	VK_CHECK(vkCreateCommandPool(device, &commandPoolInfo, NULL, &commandpool));
	VkRenderPass renderPass = 0;
	VkAttachmentDescription attachmentsrp[2] = {
	    {
	        .format = swapchaincreateinfo.imageFormat,
	        .samples = VK_SAMPLE_COUNT_1_BIT,
	        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
	        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
	        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
	        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
	    },
	    // Depth attachment (NEW)
	    {
	        .format = depthFormat,
	        .samples = VK_SAMPLE_COUNT_1_BIT,
	        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
	        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
	        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
	        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
	        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	    }};
	VkAttachmentReference colorAttachments = {
	    .attachment = 0,
	    .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	};
	VkAttachmentReference depthAttachmentRef = {// Add this
	    .attachment = 1,
	    .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

	VkSubpassDescription subpass = {
	    .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
	    .colorAttachmentCount = 1,
	    .pColorAttachments = &colorAttachments,
	    .pDepthStencilAttachment = &depthAttachmentRef};
	VkRenderPassCreateInfo rpcreateInfo = {
	    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
	    .attachmentCount = ARRAYSIZE(attachmentsrp),
	    .pAttachments = attachmentsrp,
	    .subpassCount = 1,
	    .pSubpasses = &subpass,
	};
	VK_CHECK(vkCreateRenderPass(device, &rpcreateInfo, 0, &renderPass));
	u32 swapchainimageCount = 0;
	VK_CHECK(
	    vkGetSwapchainImagesKHR(device, swapchain, &swapchainimageCount, NULL));
	VkImage* swapchainImages = malloc(swapchainimageCount * sizeof(VkImage));
	VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &swapchainimageCount,
	    swapchainImages));
	VkImageView* swapchainImageViews =
	    malloc(swapchainimageCount * sizeof(VkImageView));
	for (u32 i = 0; i < swapchainimageCount; ++i)
	{
		VkImageViewCreateInfo imageViewInfo = {
		    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		    .image = swapchainImages[i],
		    .viewType = VK_IMAGE_VIEW_TYPE_2D,
		    .format = swapchaincreateinfo.imageFormat,
		    .components =
		        {
		            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
		            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
		            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
		            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
		        },
		    .subresourceRange =
		        {
		            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		            .baseMipLevel = 0,
		            .levelCount = 1,
		            .baseArrayLayer = 0,
		            .layerCount = 1,
		        },
		};
		VK_CHECK(vkCreateImageView(device, &imageViewInfo, NULL,
		    &swapchainImageViews[i]));
	}
	VkFramebuffer framebuffers[swapchainimageCount];
	for (u32 i = 0; i < swapchainimageCount; ++i)
	{
		VkImageView attachments[2] = {swapchainImageViews[i], depthImageView};
		VkFramebufferCreateInfo framebufferInfo = {
		    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		    .renderPass = renderPass,
		    // .attachmentCount = ARRAYSIZE(attachments),
		    .attachmentCount = 2,
		    .pAttachments = attachments,
		    .width = windowWidth,
		    .height = windowHeight,
		    .layers = 1,
		};
		VK_CHECK(
		    vkCreateFramebuffer(device, &framebufferInfo, NULL, &framebuffers[i]));
	}
	VkShaderModule triangleVS = LoadShaderModule("shaders/tri.vert.spv", device);
	VkShaderModule triangleFS = LoadShaderModule("shaders/tri.frag.spv", device);

	VkPipelineLayout pipelinelayout;
	VkPipelineLayoutCreateInfo pipelinecreateInfo = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	    .setLayoutCount = 1,
	    .pSetLayouts = &descriptorSetLayout, // NEW
	};
	VK_CHECK(
	    vkCreatePipelineLayout(device, &pipelinecreateInfo, 0, &pipelinelayout));
	VkGraphicsPipelineCreateInfo pipelineinfo = {
	    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
	};
	VkPipelineShaderStageCreateInfo stages[2] = {
	    {
	        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	        .stage = VK_SHADER_STAGE_VERTEX_BIT,
	        .module = triangleVS,
	        .pName = "main",
	    },
	    {
	        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
	        .module = triangleFS,
	        .pName = "main",
	    },
	};
	pipelineinfo.stageCount = ARRAYSIZE(stages);
	pipelineinfo.pStages = stages;
	VkVertexInputBindingDescription bindingDesc = {
	    .binding = 0,
	    .stride = sizeof(Vertex),
	    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	VkVertexInputAttributeDescription attributes[] = {
	    {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, pos)},
	    {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal)},
	    {.location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, texcoord)},
	};

	VkPipelineVertexInputStateCreateInfo vertexInput = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
	    .vertexBindingDescriptionCount = 1,
	    .pVertexBindingDescriptions = &bindingDesc,
	    .vertexAttributeDescriptionCount = 3,
	    .pVertexAttributeDescriptions = attributes,
	};

	pipelineinfo.pVertexInputState = &vertexInput;
	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
	    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	pipelineinfo.pInputAssemblyState = &inputAssembly;
	VkPipelineViewportStateCreateInfo viewportState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
	    .viewportCount = 1,
	    .scissorCount = 1,
	};
	pipelineinfo.pViewportState = &viewportState;
	VkPipelineRasterizationStateCreateInfo rasterizationState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
	    .lineWidth = 1.f,
	};
	pipelineinfo.pRasterizationState = &rasterizationState;
	VkPipelineMultisampleStateCreateInfo multisampleState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
	    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	pipelineinfo.pMultisampleState = &multisampleState;
	VkPipelineDepthStencilStateCreateInfo depthStencilState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
	    .depthTestEnable = VK_TRUE,           // NEW
	    .depthWriteEnable = VK_TRUE,          // NEW
	    .depthCompareOp = VK_COMPARE_OP_LESS, // NEW
	    .depthBoundsTestEnable = VK_FALSE,
	    .minDepthBounds = 0.0f,
	    .maxDepthBounds = 1.0f,
	    .stencilTestEnable = VK_FALSE,
	};
	pipelineinfo.pDepthStencilState = &depthStencilState;
	VkPipelineColorBlendAttachmentState colorAttachmentState = {
	    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
	                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};
	VkPipelineColorBlendStateCreateInfo colorBlendState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
	    .attachmentCount = 1,
	    .pAttachments = &colorAttachmentState,
	};
	pipelineinfo.pColorBlendState = &colorBlendState;
	VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamicState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
	    .dynamicStateCount = sizeof(dynamicStates) / sizeof(dynamicStates[0]),
	    .pDynamicStates = dynamicStates,
	};
	pipelineinfo.pDynamicState = &dynamicState;
	pipelineinfo.layout = pipelinelayout;
	pipelineinfo.renderPass = renderPass;
	VkPipeline pipeline = 0;
	VkPipelineCache pipelineCache = 0;
	VK_CHECK(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineinfo, 0,
	    &pipeline));
	VkCommandBuffer commandBuffer;
	// Load OBJ model
	 fastObjMesh* mesh = fast_obj_read("monkey.obj");
	//fastObjMesh* mesh = fast_obj_read("/home/lka/myprojects/vulkantriangle/ok.obj");
	if (!mesh)
	{
		fprintf(stderr, "Failed to load OBJ file\n");
		exit(1);
	}

	//
	uint32_t index_count = mesh->index_count;
	uint32_t vertex_count = index_count; // One vertex per index

	Vertex* vertices = malloc(sizeof(Vertex) * vertex_count);
	uint32_t* indices = malloc(sizeof(uint32_t) * index_count);

	unsigned int vertex_index = 0;

	for (unsigned int i = 0; i < index_count; i++)
	{
		fastObjIndex idx = mesh->indices[i];

		// Position (required)
		if (idx.p > 0)
		{
			unsigned int p_idx = (idx.p - 1) * 3;
			vertices[i].pos[0] = mesh->positions[p_idx];
			vertices[i].pos[1] = mesh->positions[p_idx + 1];
			vertices[i].pos[2] = mesh->positions[p_idx + 2];
		}
		else
		{
			// Default position if not available
			vertices[i].pos[0] = 0.0f;
			vertices[i].pos[1] = 0.0f;
			vertices[i].pos[2] = 0.0f;
		}

		// Texture coordinates (optional)
		if (idx.t > 0 && mesh->texcoords)
		{
			unsigned int t_idx = (idx.t - 1) * 2;
			vertices[i].texcoord[0] = mesh->texcoords[t_idx];
			vertices[i].texcoord[1] = 1.0f - mesh->texcoords[t_idx + 1]; // Flip Y
		}
		else
		{
			vertices[i].texcoord[0] = 0.0f;
			vertices[i].texcoord[1] = 0.0f;
		}

		// Normals (optional)
		if (idx.n > 0 && mesh->normals)
		{
			unsigned int n_idx = (idx.n - 1) * 3;
			vertices[i].normal[0] = mesh->normals[n_idx];
			vertices[i].normal[1] = mesh->normals[n_idx + 1];
			vertices[i].normal[2] = mesh->normals[n_idx + 2];
		}
		else
		{
			// Default normal pointing up
			vertices[i].normal[0] = 0.0f;
			vertices[i].normal[1] = 1.0f;
			vertices[i].normal[2] = 0.0f;
		}

		// Create sequential indices since we're expanding vertices
		indices[i] = i;
	}

	Buffer vertexBuffer, indexBuffer;
	printf("Creating buffers for %u vertices and %u indices\n", vertex_count, index_count);
	createBuffer(device, selectedPhysicalDevice, &memprops, &vertexBuffer, vertex_count * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	memcpy(vertexBuffer.data, vertices, vertex_count * sizeof(Vertex));

	createBuffer(device, selectedPhysicalDevice, &memprops, &indexBuffer, index_count * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	memcpy(indexBuffer.data, indices, index_count * sizeof(uint32_t));
	// Add near other buffers
	Buffer uniformBuffer;
	createBuffer(device, selectedPhysicalDevice, &memprops, &uniformBuffer,
	    sizeof(mat4) * 3, // For model, view, proj matrices
	    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	// Cleanup temporary data
	fast_obj_destroy(mesh);
	printf("Loaded mesh: %u indices, %u positions, %u normals, %u texcoords\n",
	    mesh->index_count, mesh->position_count, mesh->normal_count, mesh->texcoord_count);
	// Add before rendering loop
	VkDescriptorBufferInfo bufferInfo = {
	    .buffer = uniformBuffer.vkbuffer,
	    .offset = 0,
	    .range = sizeof(mat4) * 3};

	VkWriteDescriptorSet descriptorWrite = {
	    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
	    .dstSet = descriptorSet,
	    .dstBinding = 0,
	    .dstArrayElement = 0,
	    .descriptorCount = 1,
	    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	    .pBufferInfo = &bufferInfo};

	vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, NULL);

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		// In your rendering loop, before submitting commands
		mat4 model = GLM_MAT4_IDENTITY_INIT;
		mat4 view = GLM_MAT4_IDENTITY_INIT;
		mat4 proj = GLM_MAT4_IDENTITY_INIT;

		// Move the camera back along -Z so we can see the mesh
		glm_translate(view, (vec3){0.0f, 0.0f, -3.0f}); // Camera at (0,0,3) looking at origin

		glm_perspective(glm_rad(45.0f),
		    (float)windowWidth / (float)windowHeight,
		    0.1f, 100.0f, proj);

		// GLM (cglm) produces OpenGL-style clip space, so invert Y for Vulkan
		proj[1][1] *= -1;

		unsigned char* uboBytes = (unsigned char*)uniformBuffer.data;
		memcpy(uboBytes + 0 * sizeof(mat4), &model, sizeof(mat4));
		memcpy(uboBytes + 1 * sizeof(mat4), &view, sizeof(mat4));
		memcpy(uboBytes + 2 * sizeof(mat4), &proj, sizeof(mat4));
		u32 imageIndex = 0;

		VK_CHECK(vkAcquireNextImageKHR(device, swapchain, ~0ull, imageAvailableSemaphore,
		    VK_NULL_HANDLE, &imageIndex));
		VK_CHECK(vkResetCommandPool(device, commandpool, 0));
		VkCommandBufferAllocateInfo commandBufferInfo = {
		    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		    .commandPool = commandpool,
		    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		    .commandBufferCount = 1,
		};
		VK_CHECK(
		    vkAllocateCommandBuffers(device, &commandBufferInfo, &commandBuffer));
		VkCommandBufferBeginInfo begininfo = {
		    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		VK_CHECK(vkBeginCommandBuffer(commandBuffer, &begininfo));
		VkClearValue clearValues[2] = {
		    // CHANGED from 1 to
		    // 2 to include depth
		    {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}},
		    {.depthStencil = {.depth = 1.0f, .stencil = 0}},
		};

		VkRenderPassBeginInfo renderPassBeginInfo = {
		    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		    .renderPass = renderPass,
		    .framebuffer = framebuffers[imageIndex],
		    .renderArea = {.offset = {0, 0}, .extent = {windowWidth, windowHeight}},
		    .clearValueCount = ARRAYSIZE(clearValues),
		    .pClearValues = clearValues,
		};
		vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo,
		    VK_SUBPASS_CONTENTS_INLINE);
		VkViewport viewport = {
		    .x = 0.0f,
		    .y = 0.0f,
		    .width = (float)windowWidth,
		    .height = (float)windowHeight,
		    .minDepth = 0.0f,
		    .maxDepth = 1.0f,
		};
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		VkRect2D scissor = {
		    .offset = {0, 0},
		    .extent = {windowWidth, windowHeight},
		};
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		VkDeviceSize offsets[] = {0};
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.vkbuffer, offsets);
		vkCmdBindIndexBuffer(commandBuffer, indexBuffer.vkbuffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindDescriptorSets(commandBuffer,
		    VK_PIPELINE_BIND_POINT_GRAPHICS,
		    pipelinelayout,
		    0, // First set
		    1, // Descriptor set count
		    &descriptorSet,
		    0, // Dynamic offset count
		    NULL);

		vkCmdDrawIndexed(commandBuffer, index_count, 1, 0, 0, 0);
		vkCmdEndRenderPass(commandBuffer);
		VK_CHECK(vkEndCommandBuffer(commandBuffer));
		VkSubmitInfo submitInfo = {
		    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		    .waitSemaphoreCount = 1,
		    .pWaitSemaphores = &imageAvailableSemaphore,
		    .pWaitDstStageMask =
		        (VkPipelineStageFlags[]){
		            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
		    .commandBufferCount = 1,
		    .pCommandBuffers = &commandBuffer,
		    .signalSemaphoreCount = 1,
		    .pSignalSemaphores = &renderCompleteSemaphore,
		};
		VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
		VkPresentInfoKHR presentInfo = {
		    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		    .waitSemaphoreCount = 1,
		    .pWaitSemaphores = &renderCompleteSemaphore,
		    .swapchainCount = 1,
		    .pSwapchains = &swapchain,
		    .pImageIndices = &imageIndex,
		};
		VK_CHECK(vkQueuePresentKHR(queue, &presentInfo));
		VK_CHECK(vkDeviceWaitIdle(device));
	}
	vkDeviceWaitIdle(device);
	vkFreeCommandBuffers(device, commandpool, 1, &commandBuffer);
	vkDestroyCommandPool(device, commandpool, 0);
	for (uint32_t i = 0; i < swapchainimageCount; ++i)
		vkDestroyFramebuffer(device, framebuffers[i], 0);
	for (uint32_t i = 0; i < swapchainimageCount; ++i)
		vkDestroyImageView(device, swapchainImageViews[i], 0);
	vkDestroyPipeline(device, pipeline, 0);
	vkDestroyPipelineLayout(device, pipelinelayout, 0);
	vkDestroyShaderModule(device, triangleFS, 0);
	vkDestroyShaderModule(device, triangleVS, 0);
	vkDestroyRenderPass(device, renderPass, 0);
	vkDestroySemaphore(device, renderCompleteSemaphore, 0);
	vkDestroySemaphore(device, imageAvailableSemaphore, 0);
	vkDestroySwapchainKHR(device, swapchain, 0);
	vkDestroySurfaceKHR(instance, surface, 0);
	glfwDestroyWindow(window);
	vkDestroyDevice(device, 0);
	vkDestroyInstance(instance, 0);
	return 0;
}
