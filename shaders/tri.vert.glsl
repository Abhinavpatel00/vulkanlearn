//  #version 450


// layout(location = 0) in vec3 pos;
// layout(location = 1) in vec3 normal;
// layout(location = 2) in vec2 texcoord;

// layout(location = 0) out vec4 color;

// void main()
// {
// 	gl_Position = vec4(pos + vec3(0, 0, 0.5), 1.0);

// 	color = vec4(normal * 0.5 + vec3(0.5), 1.0);
// }

#version 450
// Input
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

// Output
layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexCoord;

// Uniform buffer
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragNormal = inNormal;
    fragTexCoord = inTexCoord;
}