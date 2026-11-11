// Deliberately NO #version directive (GLSL ES 1.00). The output is gl_FragData[0] instead of
// gl_FragColor (GLSL ES 1.00 §7.2; only element 0 exists for ES 2.0-style shaders, GLSL ES 3.00 §1.5).
#if __VERSION__ != 100
#error "a shader without #version must be compiled as GLSL ES 1.00"
#endif
precision mediump float;
uniform sampler2D u_tex;
uniform int u_mode;
varying vec2 v_uv;
varying vec4 v_proj;

void main()
{
    if (u_mode == 0)
        gl_FragData[0] = texture2D(u_tex, v_uv);
    else if (u_mode == 1)
        gl_FragData[0] = texture2DProj(u_tex, v_proj.xyw);  // vec3 form: (s, t) / q
    else
        gl_FragData[0] = texture2DProj(u_tex, v_proj);      // vec4 form: (s, t) / q, third component ignored
}
