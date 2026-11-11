#version 310 es
// LOC is defined by main.cpp (inserted after the #version line) as either MAX_UNIFORM_LOCATIONS - 1, the
// highest valid explicit location, or MAX_UNIFORM_LOCATIONS, one past it. GLSL ES 3.10 section 4.4.3: "Valid
// locations for default-block uniform variable locations are in the range of 0 to the implementation-defined
// maximum number of uniform locations minus one."
precision highp float;

layout(location = LOC) uniform float u_edge;

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = vec4(u_edge, 0.25, 0.75, 1.0);
}
