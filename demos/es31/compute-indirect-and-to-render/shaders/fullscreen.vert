#version 310 es
// One large triangle covering the viewport: every pixel of the viewport is shaded exactly once (there is no
// diagonal edge inside the viewport), which the fragment-shader read-after-write passes rely on.
void main()
{
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
