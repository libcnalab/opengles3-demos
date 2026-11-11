#version 320 es
// Geometry shader (ES 3.2 section 11.3): draws the edges of the generated triangles as lines, to make the level of detail
// visible (core ES has no polygon mode).
layout(triangles) in;
layout(line_strip, max_vertices = 4) out;

void main()
{
    // Triangles with a vertex far outside the window (the terrain next to and behind the camera) are not drawn: nothing
    // useful to see, and llvmpipe / lavapipe mis-rasterize lines whose end points are that far outside the viewport (long
    // streaks across the sky), while radeonsi clips them correctly. Not part of the validated image.
    for (int i = 0; i < 3; ++i)
    {
        vec4 p = gl_in[i].gl_Position;
        if (p.w < 0.5 || abs(p.x) > 2.0 * p.w || abs(p.y) > 2.0 * p.w) return;
    }
    for (int i = 0; i < 4; ++i)
    {
        gl_Position = gl_in[i % 3].gl_Position;
        gl_Position.z -= 0.0004 * gl_Position.w;   // slightly towards the camera, so the lines win the depth test
        EmitVertex();
    }
    EndPrimitive();
}
