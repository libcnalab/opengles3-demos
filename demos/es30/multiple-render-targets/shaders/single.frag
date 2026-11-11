#version 300 es
// A fragment shader with ONE output that is explicitly bound to location 1. A single output does
// not need a location (it would default to 0), but it may have one: this color goes to draw
// buffer 1 and nowhere else. Draw buffer 0 is not written by this shader, so the application must
// map it to NONE (an unwritten fragment color would be undefined, ES 3.0 3.9.2.3).
precision highp float;

layout(location = 1) out vec4 o_only;

void main()
{
    o_only = vec4(0.8, 0.4, 0.2, 1.0);
}
