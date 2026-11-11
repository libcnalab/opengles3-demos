#version 300 es
// One POINTS vertex per instance. gl_InstanceID chooses the position, the point size and the color;
// gl_PointSize (GLSL ES 3.00 section 7.1) is the side of the square the point rasterizes to
// (ES 3.0 section 3.4). The window-space position is built so that the point centers fall on pixel
// centers; with odd sizes the square edges then lie on integer coordinates, away from every pixel center.
uniform vec2 u_cell;  // viewport size in pixels

out vec3 v_color;

void main()
{
    float i = float(gl_InstanceID);
    vec2 window = vec2(u_cell.x * (0.125 + 0.25 * i) + 0.5, u_cell.y * 0.5 + 0.5);
    gl_Position = vec4(window / u_cell * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = 3.0 + 2.0 * i;
    v_color = vec3(float(gl_InstanceID & 1), float((gl_InstanceID >> 1) & 1), 1.0 - 0.25 * i);
}
