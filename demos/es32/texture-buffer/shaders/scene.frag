#version 320 es
precision highp float;
precision highp int;

flat in vec4 v_color;
flat in uint v_material;
in vec3 v_normal;

uniform vec3 u_light;  // unit vector towards the light

out vec4 o_color;

void main()
{
    vec3 n = normalize(v_normal);
    float shade;
    if (v_material == 0u)
        shade = 0.30 + 0.70 * max(dot(n, u_light), 0.0);  // diffuse
    else if (v_material == 1u)
        shade = 1.0;                                       // unlit
    else
        shade = n.y > 0.5 ? 1.0 : 0.5;                     // bright top, dim sides
    o_color = vec4(v_color.rgb * shade, 1.0);
}
