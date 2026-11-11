#version 320 es
// One cube mesh drawn N times with glDrawElementsInstanced. All per-instance data comes from buffer textures:
//   u_transform  RGBA32F, two texels per instance: (x, y, z, scale) and (angle about Y, 0, 0, 0). Rewritten by
//                glBufferSubData on the TEXTURE_BUFFER target every frame (the animation).
//   u_color      RGBA8 (normalized): one texel per instance.
//   u_material   R16UI: one texel per instance, read through a usamplerBuffer.
// The three textures are different ranges of one buffer object, attached with glTexBufferRange.
layout(location = 0) in vec3 a_position;  // unit cube, corners at +-0.5
layout(location = 1) in vec3 a_normal;

uniform highp samplerBuffer u_transform;
uniform highp samplerBuffer u_color;
uniform highp usamplerBuffer u_material;
uniform highp mat4 u_view_proj;

out highp vec3 v_normal;
flat out highp vec4 v_color;
flat out highp uint v_material;

void main()
{
    vec4 t0 = texelFetch(u_transform, 2 * gl_InstanceID);
    vec4 t1 = texelFetch(u_transform, 2 * gl_InstanceID + 1);
    float c = cos(t1.x);
    float s = sin(t1.x);
    mat3 rot = mat3(c, 0.0, -s,   // column 0
                    0.0, 1.0, 0.0,
                    s, 0.0, c);   // rotation about +Y: x' = c x + s z, z' = -s x + c z
    vec3 world = t0.xyz + rot * (a_position * t0.w);
    v_normal = rot * a_normal;
    v_color = texelFetch(u_color, gl_InstanceID);
    v_material = texelFetch(u_material, gl_InstanceID).x;
    gl_Position = u_view_proj * vec4(world, 1.0);
}
