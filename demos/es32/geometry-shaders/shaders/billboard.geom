#version 320 es
// Cell A: amplification. One POINTS primitive in, one 4-vertex TRIANGLE_STRIP out (a camera-facing quad).
// The palette entry comes from gl_PrimitiveIDIn: the index of the point inside the draw call.
precision highp float;

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 1) uniform mat4 u_proj;

in VS_OUT
{
    float size;
} gs_in[];  // one element: the input primitive is a point

// Output interface block: captured by transform feedback as "GS_OUT.eye", "GS_OUT.uv", "GS_OUT.prim".
out GS_OUT
{
    vec3 eye;        // eye-space position of this corner
    vec2 uv;         // [-1,1]^2 across the quad
    flat vec3 color; // palette entry of this particle
    flat int prim;   // copy of gl_PrimitiveIDIn
} gs_out;

const vec3 kPalette[6] = vec3[6](vec3(1.00, 0.35, 0.30), vec3(1.00, 0.80, 0.25), vec3(0.45, 0.90, 0.40),
                                 vec3(0.30, 0.80, 0.95), vec3(0.50, 0.45, 1.00), vec3(0.95, 0.40, 0.85));

void main()
{
    vec3 center = gl_in[0].gl_Position.xyz;
    float s = gs_in[0].size;
    vec3 color = kPalette[gl_PrimitiveIDIn % 6];
    for (int i = 0; i < 4; ++i)
    {
        // Strip order: (-1,-1) (+1,-1) (-1,+1) (+1,+1) -> triangles (0,1,2) and (2,1,3).
        vec2 corner = vec2(float(i & 1) * 2.0 - 1.0, float(i >> 1) * 2.0 - 1.0);
        vec3 p = center + vec3(corner * s, 0.0);
        gs_out.eye = p;
        gs_out.uv = corner;
        gs_out.color = color;
        gs_out.prim = gl_PrimitiveIDIn;
        gl_Position = u_proj * vec4(p, 1.0);
        EmitVertex();
    }
    EndPrimitive();
}
