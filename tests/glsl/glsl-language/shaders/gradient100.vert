#version 100
// GLSL ES 1.00 vertex shader: per-vertex inputs use the attribute qualifier (§4.3.3), outputs to the
// fragment shader use varying (§4.3.5). Attribute locations are bound with glBindAttribLocation.
attribute vec2 a_position;
attribute vec3 a_color;
varying vec3 v_color;

void main()
{
    v_color = a_color;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
