#version 300 es
// Same flat color, but the fragment shader replaces the fragment's depth (GLSL ES 3.00 §7.2).
//
//   gl_FragDepth = u_depth_line.x + u_depth_line.y * gl_FragCoord.x
//
// A slope of 0 writes a constant depth; a non-zero slope writes a depth that ramps across the
// window, i.e. a different depth for every fragment of the same primitive. In both cases the
// depth that was interpolated from the vertices is thrown away. The shader writes gl_FragDepth
// unconditionally, as the specification requires once it is statically assigned anywhere.
precision highp float;
uniform vec4 u_color;
uniform vec2 u_depth_line;
layout(location = 0) out vec4 o_color;

void main()
{
    o_color = u_color;
    gl_FragDepth = u_depth_line.x + u_depth_line.y * gl_FragCoord.x;
}
