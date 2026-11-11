#version 320 es
// Tessellation evaluation shader (ES 3.2 section 11.2.3, GLSL ES 3.20 section 4.4.1 "Tessellation Evaluation Inputs").
//
// The application #defines TESS_LAYOUT right after the #version line, for example
//     #define TESS_LAYOUT quads, fractional_odd_spacing, cw, point_mode
// and exactly one of DOMAIN_TRIANGLES / DOMAIN_QUADS / DOMAIN_ISOLINES. The input layout selects the primitive generator's
// domain, the edge spacing, the winding of generated triangles and whether it emits points.
layout(TESS_LAYOUT) in;

in float t_tag[];          // per-vertex inputs: the TCS output patch
in vec3 t_pos[];
patch in vec4 p_center;    // per-patch input written by the TCS

uniform mat4 u_mvp;
uniform float u_dome;      // height of the paraboloid bump the patch is bent into (0 = flat)
uniform float u_radius;

out float v_h;             // height of the vertex, for shading
out vec3 v_n;              // normal of the bump

// Captured with transform feedback (the TES is the last vertex processing stage, ES 3.2 section 12.1).
out vec4 tf_coord;         // gl_TessCoord, gl_PrimitiveID
out vec4 tf_outer;         // gl_TessLevelOuter exactly as the TCS wrote it (before clamping and rounding)
out vec4 tf_misc;          // gl_TessLevelInner[0..1], gl_PatchVerticesIn (= TCS output patch size), t_tag[0]
out vec4 tf_center;        // p_center

void main()
{
#if defined(DOMAIN_TRIANGLES)
    // gl_TessCoord is a barycentric coordinate: u weighs patch vertex 0, v vertex 1, w vertex 2.
    vec3 p = gl_TessCoord.x * gl_in[0].gl_Position.xyz + gl_TessCoord.y * gl_in[1].gl_Position.xyz +
             gl_TessCoord.z * gl_in[2].gl_Position.xyz;
#else
    // Quads and isolines: (u, v) with w = 0; bilinear interpolation of four corners
    // (0,0) = vertex 0, (1,0) = vertex 1, (0,1) = vertex 2, (1,1) = vertex 3.
    vec3 p = mix(mix(gl_in[0].gl_Position.xyz, gl_in[1].gl_Position.xyz, gl_TessCoord.x),
                 mix(gl_in[2].gl_Position.xyz, gl_in[3].gl_Position.xyz, gl_TessCoord.x), gl_TessCoord.y);
#endif
    vec2 d = (p.xy - p_center.xy) / u_radius;
    float t = max(0.0, 1.0 - dot(d, d));
    float h = u_dome * t * t;
    v_h = h;
    // Normal of the bump h = D t^2, t = 1 - |d|^2: dh/dx = -4 D t d.x / R (zero where t is clamped).
    vec2 grad = t > 0.0 ? -4.0 * u_dome * t * d / u_radius : vec2(0.0);
    v_n = normalize(vec3(-grad, 1.0));
    gl_Position = u_mvp * vec4(p.xy, p.z + h, 1.0);

    tf_coord = vec4(gl_TessCoord, float(gl_PrimitiveID));
    tf_outer = vec4(gl_TessLevelOuter[0], gl_TessLevelOuter[1], gl_TessLevelOuter[2], gl_TessLevelOuter[3]);
    tf_misc = vec4(gl_TessLevelInner[0], gl_TessLevelInner[1], float(gl_PatchVerticesIn), t_tag[0]);
    tf_center = p_center;
}
