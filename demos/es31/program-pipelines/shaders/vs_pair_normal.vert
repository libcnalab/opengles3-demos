#version 310 es
// A separable VERTEX program. In the pipeline demo it is the stage that is swapped at run time.
//
// Interface towards the fragment program: matched by LOCATION (ES 3.2 section 7.4.1, GLSL ES 3.10
// section 9.2.2), not by name. There is no default-block uniform shared with the fragment stage:
// every program has its own uniform name space (GLSL ES 3.10 section 9), so "location 0" here and
// "location 0" in the other vertex programs are different variables.
layout(location = 0) uniform vec3 u_tint;   // forwarded to the fragment stage

layout(location = 0) out vec2 v_uv;         // 0..1 across the viewport
layout(location = 1) out vec3 v_tint;

void main()
{
    // No vertex attributes: one triangle that covers the viewport, derived from gl_VertexID.
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));  // (0,0) (2,0) (0,2)
    v_uv = corner;
    v_tint = u_tint;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
