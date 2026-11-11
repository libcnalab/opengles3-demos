#version 320 es
// Single-pass cube map rendering (ES 3.2 section 11.3.4.5: "project and render a scene onto all six faces of a
// cubemap texture in one pass"). layout(invocations = 6) runs the shader once per cube face for every input
// triangle; invocation k renders with the view-projection of face k and writes gl_Layer = k. For a cube map attached
// as a layered image, layer k is the face TEXTURE_CUBE_MAP_POSITIVE_X + k (Table 8.24):
// 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z.
precision highp float;

layout(triangles, invocations = 6) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) uniform mat4 u_viewproj[6];

in VS_OUT
{
    vec3 color;
} gs_in[];

out GS_OUT
{
    flat vec3 color;
} gs_out;

void main()
{
    for (int i = 0; i < 3; ++i)
    {
        gs_out.color = gs_in[i].color;
        gl_Position = u_viewproj[gl_InvocationID] * gl_in[i].gl_Position;
        gl_Layer = gl_InvocationID;
        EmitVertex();
    }
    EndPrimitive();
}
