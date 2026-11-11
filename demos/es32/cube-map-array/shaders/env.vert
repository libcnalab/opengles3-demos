#version 320 es
// Five unit-sphere meshes drawn with ONE glDrawElementsInstanced call; instance k is centred at u_sphere[k].xyz
// with radius u_sphere[k].w and reflects layer k of the cube map array.
layout(location = 0) in vec3 a_pos;  // unit sphere: position == normal

uniform highp mat4 u_view_proj;
uniform highp vec4 u_sphere[5];

out highp vec3 v_world;
out highp vec3 v_normal;
flat out highp int v_layer;

void main()
{
    vec4 s = u_sphere[gl_InstanceID];
    vec3 world = s.xyz + a_pos * s.w;
    v_world = world;
    v_normal = a_pos;
    v_layer = gl_InstanceID;
    gl_Position = u_view_proj * vec4(world, 1.0);
}
