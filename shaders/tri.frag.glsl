#version 450
// Input
layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;

// Output
layout(location = 0) out vec4 outColor;

void main() {
    // Simple Lambertian shading
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diff = max(dot(normalize(fragNormal), lightDir), 0.0);
    outColor = vec4(vec3(diff), 1.0);
}