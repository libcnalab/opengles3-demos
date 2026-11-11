#version 300 es
// A rectangle without any vertex attributes. The four corners of a triangle strip are derived
// from gl_VertexID (GLSL ES 3.00 built-in): ids 0..3 give the corners (0,0) (1,0) (0,1) (1,1).
// u_rect holds the rectangle in normalized device coordinates as (x0, y0, x1, y1), so the
// application can place every cell and every disc by changing one uniform.
uniform vec4 u_rect;

out vec2 v_local;  // -1..1 across the rectangle; the fragment shader uses it to cut out discs

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    v_local = corner * 2.0 - 1.0;
    gl_Position = vec4(mix(u_rect.xy, u_rect.zw, corner), 0.0, 1.0);
}
