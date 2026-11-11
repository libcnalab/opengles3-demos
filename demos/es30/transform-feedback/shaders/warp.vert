#version 300 es
// Vertex processing done once and recorded: a mesh vertex is displaced by a polynomial warp and both the
// warped position and the color are captured (INTERLEAVED_ATTRIBS: 5 floats = 20 bytes per vertex). A
// later pass draws the captured triangles without running this shader again.
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec3 a_color;

out vec2 v_pos;
out vec3 v_color;

void main()
{
    vec2 p = a_pos;
    p += vec2(0.25 * p.y * p.y, -0.125 * p.x * p.x);  // both terms use the original p
    v_pos = p;
    v_color = a_color;
    gl_Position = vec4(p, 0.0, 1.0);
}
