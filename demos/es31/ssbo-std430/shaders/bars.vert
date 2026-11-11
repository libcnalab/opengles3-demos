#version 310 es
// A VERTEX shader that reads a shader storage buffer: one instance per element of the runtime-sized array
// values[]. The bar width comes from values.length(), so the bars tile the viewport exactly only if length()
// is what the application expects.
precision highp float;
precision highp int;

layout(std430, binding = 0) readonly buffer Samples
{
    uint  n_samples;
    uint  tag;
    float values[];
} s;

uniform float u_height_scale;     // fraction of the viewport height per unit of value

flat out vec3 v_color;

void main()
{
    int   n   = s.values.length();
    int   bar = gl_InstanceID;
    float v   = s.values[bar];
    float w   = 2.0 / float(n);
    float x0  = -1.0 + float(bar) * w;
    float y1  = -1.0 + 2.0 * v * u_height_scale;
    float x   = (gl_VertexID & 1) == 0 ? x0 : x0 + w;      // strip: (x0,-1) (x1,-1) (x0,y1) (x1,y1)
    float y   = (gl_VertexID & 2) == 0 ? -1.0 : y1;
    gl_Position = vec4(x, y, 0.0, 1.0);
    v_color     = vec3(v / 13.0, 1.0 - v / 13.0, 0.4);
}
