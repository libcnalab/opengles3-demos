// No #version line: attribute/varying/gl_FragColor only exist in GLSL ES 1.00.
precision mediump float;
varying vec4 v_color;
void main() { gl_FragColor = v_color; }
