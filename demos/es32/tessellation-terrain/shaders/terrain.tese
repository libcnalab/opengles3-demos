#version 320 es
// Tessellation evaluation shader: places every generated vertex on the terrain (displacement mapping). The application inserts
// terrain_common.glsl after the #version line.
//
// fractional_odd_spacing makes the mesh change continuously with the level: a new pair of vertices grows out of the middle of an
// edge instead of the whole edge popping to a finer subdivision. ccw makes the generated triangles front-facing.
layout(quads, fractional_odd_spacing, ccw) in;

uniform mat4 u_viewproj;

patch in float p_level;

out vec3 v_world;
flat out float v_level;

// Captured with transform feedback by the validation (the TES is the last vertex processing stage).
out vec4 tf_pos;     // world position, gl_PrimitiveID
out vec4 tf_uv;      // gl_TessCoord.xy, p_level, 0
out vec4 tf_outer;   // gl_TessLevelOuter as written by the TCS
out vec4 tf_inner;   // gl_TessLevelInner, 0, 0

void main()
{
    vec2 xz = mix(mix(gl_in[0].gl_Position.xz, gl_in[1].gl_Position.xz, gl_TessCoord.x),
                  mix(gl_in[2].gl_Position.xz, gl_in[3].gl_Position.xz, gl_TessCoord.x), gl_TessCoord.y);
    vec3 world = vec3(xz.x, height_at(xz), xz.y);   // sample the height map: the vertex is displaced
    gl_Position = u_viewproj * vec4(world, 1.0);
    v_world = world;
    v_level = p_level;

    tf_pos = vec4(world, float(gl_PrimitiveID));
    tf_uv = vec4(gl_TessCoord.xy, p_level, 0.0);
    tf_outer = vec4(gl_TessLevelOuter[0], gl_TessLevelOuter[1], gl_TessLevelOuter[2], gl_TessLevelOuter[3]);
    tf_inner = vec4(gl_TessLevelInner[0], gl_TessLevelInner[1], 0.0, 0.0);
}
