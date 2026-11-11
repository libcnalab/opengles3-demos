#version 320 es
layout(triangles, fractional_odd_spacing, ccw) in;
patch in float p_scale;
void main()
{
    gl_Position = p_scale * (gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position +
                             gl_TessCoord.z * gl_in[2].gl_Position);
}
