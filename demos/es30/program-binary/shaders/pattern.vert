#version 300 es
// A full-viewport triangle. a_pos has no layout qualifier on purpose: the application binds it to
// attribute 5 with glBindAttribLocation before the link, and the demo checks that a program loaded
// with glProgramBinary restores exactly that assignment (ES 3.0 §2.12.4).

in vec2 a_pos;

uniform vec2 u_scale;  // (1, 1) when set by the application; reset to (0, 0) by glProgramBinary

void main()
{
    gl_Position = vec4(a_pos * u_scale, 0.0, 1.0);
}
