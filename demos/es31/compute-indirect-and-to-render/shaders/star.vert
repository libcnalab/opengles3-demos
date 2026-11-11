#version 310 es
// Vertex stage of the compute-generated star: position (float) and sector (unsigned integer attribute) come
// straight from the buffer that gen_geometry.comp filled.
layout(location = 0) in vec2 a_pos;
layout(location = 1) in uint a_sector;
flat out vec3 v_color;

vec3 palette(uint s)
{
    return 0.25 + 0.75 * vec3(float(s & 3u) / 3.0, float((s >> 1u) & 3u) / 3.0, float((s >> 2u) & 3u) / 3.0);
}

void main()
{
    v_color = palette(a_sector);
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
