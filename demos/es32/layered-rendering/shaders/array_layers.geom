#version 320 es
// 2D array texture, 64x64x4. Four POINTS are drawn; the layer of point n is n = gl_PrimitiveIDIn, so ONE draw call
// paints four different images. Every primitive is a pixel-aligned rectangle (coordinates are integers divided by 32),
// so the rasterized result is exact. Layer n shows
//   - a background of its own colour,
//   - a body rectangle [8+4n, 56-4n]^2 that shrinks with n,
//   - n+1 white 4x4 squares along the bottom edge (the layer index as a count).
precision highp float;

layout(points) in;
layout(triangle_strip, max_vertices = 24) out;

flat out vec4 v_color;

const vec4 kBack[4] = vec4[4](vec4(0.2, 0.0, 0.0, 1.0), vec4(0.0, 0.2, 0.0, 1.0), vec4(0.0, 0.0, 0.4, 1.0), vec4(0.2, 0.0, 0.2, 1.0));
const vec4 kBody[4] = vec4[4](vec4(1.0, 0.4, 0.4, 1.0), vec4(0.4, 1.0, 0.4, 1.0), vec4(0.4, 0.4, 1.0, 1.0), vec4(1.0, 0.8, 0.4, 1.0));

// One rectangle (pixel coordinates of the 64x64 layer) as its own 4-vertex strip, assigned to `layer`.
void rect(float x0, float y0, float x1, float y1, vec4 color, int layer)
{
    v_color = color;
    gl_Layer = layer;  // the same layer for all four vertices: defined behaviour (ES 3.2 section 9.8)
    gl_Position = vec4(x0 / 32.0 - 1.0, y0 / 32.0 - 1.0, 0.0, 1.0); EmitVertex();
    gl_Position = vec4(x1 / 32.0 - 1.0, y0 / 32.0 - 1.0, 0.0, 1.0); EmitVertex();
    gl_Position = vec4(x0 / 32.0 - 1.0, y1 / 32.0 - 1.0, 0.0, 1.0); EmitVertex();
    gl_Position = vec4(x1 / 32.0 - 1.0, y1 / 32.0 - 1.0, 0.0, 1.0); EmitVertex();
    EndPrimitive();
}

void main()
{
    int n = gl_PrimitiveIDIn;
    float f = float(n);
    rect(0.0, 0.0, 64.0, 64.0, kBack[n], n);
    rect(8.0 + 4.0 * f, 8.0 + 4.0 * f, 56.0 - 4.0 * f, 56.0 - 4.0 * f, kBody[n], n);
    for (int i = 0; i <= n; ++i)
        rect(4.0 + 8.0 * float(i), 2.0, 8.0 + 8.0 * float(i), 6.0, vec4(1.0), n);
}
