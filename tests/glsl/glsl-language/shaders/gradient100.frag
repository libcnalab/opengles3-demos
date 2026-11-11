#version 100
// GLSL ES 1.00 fragment shader: reads the interpolated varying and writes the built-in gl_FragColor (§7.2).
precision mediump float;
varying vec3 v_color;

void main()
{
    gl_FragColor = vec4(v_color, 1.0);
}
