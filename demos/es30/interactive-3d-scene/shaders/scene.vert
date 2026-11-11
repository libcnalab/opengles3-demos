#version 300 es
// Main pass, vertex stage. One shader for every object of the scene:
//
//   locations 0..4  per-vertex data of the mesh (divisor 0)
//   location 5      a_instance = (offset.xyz, uniform scale) of the instance
//   location 6      a_tint     = colour multiplier of the instance
//
// Instanced objects (pillars, tree trunks, tree canopies) source locations 5/6 from an instance
// buffer with glVertexAttribDivisor(loc, 1) (ES 3.0 2.9.3): the element is floor(instance / 1).
// Ordinary objects leave those arrays DISABLED in their VAO, so the shader receives the current
// generic attribute values (ES 3.0 2.8) that the demo set once with glVertexAttrib4f(5, 0, 0, 0, 1)
// and glVertexAttrib3f(6, 1, 1, 1): offset 0, scale 1, no tint.
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_tangent;   // xyz = dP/du, w = handedness of the UV frame
layout(location = 4) in vec3 a_bary;      // (1,0,0), (0,1,0), (0,0,1) at the corners of each triangle
layout(location = 5) in vec4 a_instance;
layout(location = 6) in vec3 a_tint;

uniform mat4 u_model;          // the prototype's placement (translation + rotation about Y)
uniform mat3 u_normal_matrix;  // inverse transpose of u_model's upper 3x3
uniform mat4 u_view_proj;      // camera: perspective * look-at
uniform mat4 u_light_vp;       // sun: orthographic * look-at (the shadow pass uses the same matrix)

out vec3 v_world;
out vec3 v_normal;
out vec3 v_tangent;
out vec3 v_bitangent;
out vec2 v_uv;
out vec3 v_bary;
out vec3 v_tint;
out vec4 v_light;  // clip-space position as the sun sees it: the shadow-map lookup coordinate

void main()
{
    // The instance transform is a uniform scale plus a translation, applied after the model matrix.
    // A positive uniform scale does not change directions, so normals and tangents ignore it.
    vec3 world = a_instance.xyz + a_instance.w * (u_model * vec4(a_position, 1.0)).xyz;

    vec3 n = normalize(u_normal_matrix * a_normal);
    vec3 t = normalize(mat3(u_model) * a_tangent.xyz);
    t = normalize(t - dot(t, n) * n);  // Gram-Schmidt (es30-normal-mapping)

    v_world = world;
    v_normal = n;
    v_tangent = t;
    v_bitangent = cross(n, t) * a_tangent.w;
    v_uv = a_uv;
    v_bary = a_bary;
    v_tint = a_tint;
    v_light = u_light_vp * vec4(world, 1.0);
    gl_Position = u_view_proj * vec4(world, 1.0);
}
