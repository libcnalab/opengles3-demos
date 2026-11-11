#version 310 es
// imageStore in a FRAGMENT shader (needs MAX_FRAGMENT_IMAGE_UNIFORMS > 0; ES 3.1 allows 0, ES 3.2 requires 4).
// Each covered pixel is shaded exactly once and writes a value that depends on its position.
precision highp float;
layout(r32ui, binding = 0) uniform writeonly highp uimage2D u_out;
layout(location = 0) out vec4 o_color;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    imageStore(u_out, p, uvec4(uint(p.x * 3 + p.y * 5 + 1)));
    o_color = vec4(0.0);
}
