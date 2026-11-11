#version 320 es
// Concentric rings that become thinner than a pixel towards the edge. Multisampling alone does not antialias this
// pattern, because the shader runs once per pixel; sample shading does, because the shader then runs per sample.
// v_pos is an ordinary input (no "sample" qualifier), so only SAMPLE_SHADING with MinSampleShading(1.0) makes it
// per sample (ES 3.2 section 13.4.1).
precision highp float;

in highp vec2 v_pos;
uniform vec2 u_center;

out vec4 o_color;

void main()
{
    vec2 d = v_pos - u_center;
    float g = fract(dot(d, d) / 48.0);
    o_color = g < 0.5 ? vec4(0.98, 0.80, 0.25, 1.0) : vec4(0.10, 0.25, 0.55, 1.0);
}
