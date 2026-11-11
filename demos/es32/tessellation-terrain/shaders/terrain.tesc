#version 320 es
// Tessellation control shader: dynamic level of detail. The application inserts terrain_common.glsl after the #version line
// and, for the bounding-box variant, #define WRITE_BOUNDING_BOX.
//
// A patch is subdivided more the closer it is to the camera: level = clamp(u_lod_scale / distance, 1, u_max_level). The level of
// an EDGE depends only on the two corners of that edge, computed in the same order by both patches that share it, so
// neighbouring patches always get identical outer levels and no cracks open between them (inner levels are free).
layout(vertices = 4) out;

uniform mat4 u_viewproj;
uniform vec3 u_lod_eye;        // the point the distance is measured from (the camera, or a frozen copy of it)
uniform float u_lod_scale;
uniform float u_max_level;
uniform vec2 u_height_range;   // lowest and highest height of the whole terrain

patch out float p_level;       // mean outer level, for the LOD tint in the fragment shader

float edge_level(vec2 a, vec2 b)
{
    vec2 m = 0.5 * (a + b);
    vec3 mid = vec3(m.x, height_at(m), m.y);   // the middle of the displaced edge (the TCS can fetch textures too)
    return clamp(u_lod_scale / distance(u_lod_eye, mid), 1.0, u_max_level);
}

void main()
{
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

    vec2 c0 = gl_in[0].gl_Position.xz;   // (u, v) = (0, 0)
    vec2 c1 = gl_in[1].gl_Position.xz;   // (1, 0)
    vec2 c2 = gl_in[2].gl_Position.xz;   // (0, 1)
    vec2 c3 = gl_in[3].gl_Position.xz;   // (1, 1)

    if (gl_InvocationID == 0)
    {
        float left = edge_level(c0, c2);     // u = 0 edge: gl_TessLevelOuter[0]
        float bottom = edge_level(c0, c1);   // v = 0 edge: [1]
        float right = edge_level(c1, c3);    // u = 1 edge: [2]
        float top = edge_level(c2, c3);      // v = 1 edge: [3]
        gl_TessLevelOuter[0] = left;
        gl_TessLevelOuter[1] = bottom;
        gl_TessLevelOuter[2] = right;
        gl_TessLevelOuter[3] = top;
        gl_TessLevelInner[0] = max(bottom, top);   // segments along u (the horizontal edges, Figure 11.3)
        gl_TessLevelInner[1] = max(left, right);   // segments along v
        p_level = 0.25 * (left + bottom + right + top);
    }

#ifdef WRITE_BOUNDING_BOX
    // The primitive bounding box (ES 3.2 section 13.2): the clip-space extent of everything this patch can generate. The patch
    // is contained in the world-space box [x0,x1] x [lowest,highest] x [z0,z1]; the clip-space coordinates of its 8 corners
    // bound the clip coordinates of every point inside (clip space is a linear image of world space). The minimum and the
    // maximum of each of x, y, z, w over the corners are a correct box. The box only lets an implementation skip work: the image
    // must not change. Every invocation writes it (an invocation that does not write it leaves its components undefined).
    vec2 lo2 = min(min(c0, c1), min(c2, c3));
    vec2 hi2 = max(max(c0, c1), max(c2, c3));
    vec4 lo = vec4(1e30);
    vec4 hi = vec4(-1e30);
    for (int k = 0; k < 8; ++k)
    {
        vec3 p = vec3(((k & 1) == 0) ? lo2.x : hi2.x, ((k & 2) == 0) ? u_height_range.x : u_height_range.y,
                      ((k & 4) == 0) ? lo2.y : hi2.y);
        vec4 clip = u_viewproj * vec4(p, 1.0);
        lo = min(lo, clip);
        hi = max(hi, clip);
    }
    gl_BoundingBox[0] = lo;
    gl_BoundingBox[1] = hi;
#endif
}
