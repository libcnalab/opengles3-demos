#version 300 es
// One program for every lookup function whose level of detail is NOT (only) a consequence of the
// screen-space mapping: textureLod, textureGrad, textureProjLod, textureSize, texelFetch, plus the
// implicit-LOD texture() / textureProj() as a baseline. The function is chosen by a uniform, so all
// branches are dynamically uniform (implicit derivatives stay defined, GLSL ES 3.00 8.8).
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_tex;
uniform int u_func;
uniform float u_lod;        // explicit level of detail / level index / lod argument of textureSize
uniform vec2 u_dx;          // textureGrad: d(s,t)/dx
uniform vec2 u_dy;          // textureGrad: d(s,t)/dy
uniform float u_q;          // projective coordinate q (the last component of P)
uniform ivec2 u_origin;     // lower left window pixel of the viewport (texelFetch coordinates)

in vec2 v_corner;
out vec4 o_color;

void main()
{
    vec2 uv = v_corner;
    vec4 c = vec4(1.0, 0.0, 1.0, 1.0);   // magenta: an unknown u_func
    if (u_func == 0)
        c = texture(u_tex, uv);
    else if (u_func == 1)
        c = textureLod(u_tex, uv, u_lod);
    else if (u_func == 2)
        c = textureGrad(u_tex, uv, u_dx, u_dy);
    else if (u_func == 3)
        c = textureProjLod(u_tex, vec3(uv * u_q, u_q), u_lod);            // vec3 form: (s, t, q)
    else if (u_func == 4)
        c = textureProjLod(u_tex, vec4(uv * u_q, 99.0, u_q), u_lod);      // vec4 form: p is ignored for a 2D sampler
    else if (u_func == 5)
        c = textureProj(u_tex, vec3(uv * u_q, u_q));
    else if (u_func == 6)
        c = textureProj(u_tex, vec4(uv * u_q, 99.0, u_q));
    else if (u_func == 7)
        c = vec4(vec2(textureSize(u_tex, int(u_lod))) / 255.0, 0.0, 1.0);  // size encoded as bytes (w, h, 0, 255)
    else if (u_func == 8)
        c = texelFetch(u_tex, ivec2(gl_FragCoord.xy) - u_origin, int(u_lod));
    o_color = c;
}
