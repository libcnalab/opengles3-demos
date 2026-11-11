#version 310 es
// Used only to attach a vertex shader to a program next to a compute shader: such a program must fail to link
// (ES 3.2 section 7.3, "program contains objects to form a compute shader ... and program also contains objects to
// form any other type of shader").
void main()
{
    gl_Position = vec4(0.0);
}
