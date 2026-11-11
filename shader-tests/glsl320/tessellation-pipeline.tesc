#version 320 es
layout(vertices = 3) out;
patch out float p_scale;
void main()
{
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    if (gl_InvocationID == 0)
    {
        gl_TessLevelOuter[0] = 2.0; gl_TessLevelOuter[1] = 2.0; gl_TessLevelOuter[2] = 2.0;
        gl_TessLevelInner[0] = 3.0;
        p_scale = 1.0;
    }
}
