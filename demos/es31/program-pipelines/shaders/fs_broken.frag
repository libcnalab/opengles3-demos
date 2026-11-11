#version 310 es
// DELIBERATELY BROKEN: glCreateShaderProgramv must return a program name whose LINK_STATUS is FALSE
// and whose info log carries this shader's compile log (ES 3.2 section 7.3). `no_such_variable` is
// not declared.
precision highp float;
layout(location = 0) out vec4 o_color;
void main()
{
    o_color = vec4(no_such_variable, 0.0, 0.0, 1.0);
}
