#version 300 es
// Tangent-space frame construction for normal mapping (vertex stage).
//
// The vertex carries a unit tangent T = dP/du (computed on the CPU from the UV layout) and a
// handedness sign w = +-1. The bitangent is NOT stored: B = cross(N, T) * w, where w = -1 on
// surfaces whose texture is mirrored (the UV basis is left-handed relative to the surface normal).
//
// T, B and N are brought to WORLD space and passed on as three varyings; the fragment stage
// builds the matrix TBN = [T B N] (columns) from them and lights in world space.
//   T  : transforms like a position difference  -> u_model (upper 3x3)
//   N  : transforms like a normal               -> u_normal_matrix (inverse transpose)
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_tangent;  // xyz = T, w = handedness

uniform mat4 u_model;
uniform mat3 u_normal_matrix;
uniform mat4 u_view_proj;
uniform int u_use_handedness;  // 1: B = cross(N, T) * w (correct); 0: B = cross(N, T) (wrong on mirrored UVs)

out vec3 v_position;  // world space
out vec2 v_uv;
out vec3 v_tangent;
out vec3 v_bitangent;
out vec3 v_normal;

void main()
{
    vec4 world = u_model * vec4(a_position, 1.0);

    vec3 n = normalize(u_normal_matrix * a_normal);
    vec3 t = normalize(mat3(u_model) * a_tangent.xyz);
    t = normalize(t - dot(t, n) * n);  // Gram-Schmidt: keep the frame orthonormal under any model matrix
    float handedness = (u_use_handedness != 0) ? a_tangent.w : 1.0;
    vec3 b = cross(n, t) * handedness;

    v_position = world.xyz;
    v_uv = a_uv;
    v_tangent = t;
    v_bitangent = b;
    v_normal = n;
    gl_Position = u_view_proj * world;
}
