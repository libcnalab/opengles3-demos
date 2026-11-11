#version 320 es
// Per-pixel lighting of the terrain. The application inserts terrain_common.glsl after the #version line. The normal comes
// from the height map at the fragment's world position, not from the mesh, so the lighting is the same at every level of
// detail; the level only changes the silhouette and how closely the surface follows the heights.
in vec3 v_world;
flat in float v_level;

uniform vec3 u_cam_eye;
uniform float u_tint;        // 1: tint the terrain by tessellation level (blue = coarse, red = fine)
uniform float u_max_level;

out vec4 o_color;

void main()
{
    vec3 c = terrain_shade(v_world, u_cam_eye);
    vec3 ramp = mix(vec3(0.10, 0.30, 1.00), vec3(1.00, 0.20, 0.10), clamp((v_level - 1.0) / (u_max_level - 1.0), 0.0, 1.0));
    o_color = vec4(mix(c, ramp, 0.55 * u_tint), 1.0);
}
