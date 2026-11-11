#version 320 es
// 3D texture, 64x64x4. ONE point is drawn; layout(invocations = 4) runs the shader four times and gl_InvocationID is
// the slice (z) to render into. Slice k shows a background that moves from blue to green with k, a 12x12 block
// that walks along the diagonal, and a bottom bar that grows by 16 pixels per slice. All edges are on pixel
// boundaries (exact).
precision highp float;

layout(points, invocations = 4) in;
layout(triangle_strip, max_vertices = 12) out;

flat out vec4 v_color;

void rect(float x0, float y0, float x1, float y1, vec4 color, int layer)
{
    v_color = color;
    gl_Layer = layer;
    gl_Position = vec4(x0 / 32.0 - 1.0, y0 / 32.0 - 1.0, 0.0, 1.0); EmitVertex();
    gl_Position = vec4(x1 / 32.0 - 1.0, y0 / 32.0 - 1.0, 0.0, 1.0); EmitVertex();
    gl_Position = vec4(x0 / 32.0 - 1.0, y1 / 32.0 - 1.0, 0.0, 1.0); EmitVertex();
    gl_Position = vec4(x1 / 32.0 - 1.0, y1 / 32.0 - 1.0, 0.0, 1.0); EmitVertex();
    EndPrimitive();
}

void main()
{
    int k = gl_InvocationID;
    float f = float(k);
    rect(0.0, 0.0, 64.0, 64.0, vec4(0.0, 0.2 + 0.2 * f, 0.8 - 0.2 * f, 1.0), k);
    rect(6.0 + 14.0 * f, 6.0 + 14.0 * f, 18.0 + 14.0 * f, 18.0 + 14.0 * f, vec4(1.0, 1.0, 0.4, 1.0), k);
    rect(0.0, 0.0, 16.0 * (f + 1.0), 4.0, vec4(1.0, 0.6, 0.0, 1.0), k);
}
