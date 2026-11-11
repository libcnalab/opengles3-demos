#version 300 es
// Geometry pass: writes the surface description, NOT a lit color, into three render targets.
// The fragment outputs are bound to draw buffers by layout(location = N) (GLSL ES 3.00 4.3.8.2) and the
// application routes them with glDrawBuffers (ES 3.0 4.2.1):
//
//   location 0  RGBA8      albedo.rgb, specular strength in alpha
//   location 1  RGB10_A2   world-space normal encoded as n * 0.5 + 0.5 (10 bits per component), alpha = 1
//   location 2  RG8        shininess / 255 in red, emissive / 3 in green
//
// Depth is the fourth G-buffer plane; it needs no output because the depth buffer is a texture attachment.
precision highp float;

in vec3 v_position;
in vec3 v_normal;

uniform vec3 u_albedo;       // base color
uniform vec3 u_albedo2;      // second checker color / color of the sides of the floor slab
uniform float u_specular;    // specular strength  [0, 1]
uniform float u_shininess;   // Blinn-Phong exponent [1, 255]
uniform float u_emissive;    // emission scale [0, 3]: the surface adds albedo * u_emissive
uniform int u_checker;       // 1: the upward-facing surface alternates u_albedo / u_albedo2 on a 2 x 2 world-unit grid

layout(location = 0) out vec4 o_albedo;
layout(location = 1) out vec4 o_normal;
layout(location = 2) out vec4 o_material;

void main()
{
    vec3 n = normalize(v_normal);

    vec3 albedo = u_albedo;
    if (u_checker != 0)
    {
        float cell = floor(v_position.x * 0.5) + floor(v_position.z * 0.5);
        albedo = (n.y > 0.5 && mod(cell, 2.0) < 0.5) ? u_albedo : u_albedo2;
    }

    o_albedo = vec4(albedo, u_specular);
    o_normal = vec4(n * 0.5 + 0.5, 1.0);
    o_material = vec4(u_shininess / 255.0, u_emissive / 3.0, 0.0, 1.0);
}
