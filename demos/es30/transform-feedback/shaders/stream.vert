#version 300 es
// Generates 4 x 3 lattice points from gl_VertexID alone (no vertex attributes). The three outputs are
// captured in SEPARATE_ATTRIBS mode: v_pos -> binding point 0, v_color -> 1, v_size -> 2, each tightly
// packed in its own region (a range of one buffer, or a buffer of its own).
out vec2 v_pos;
out vec3 v_color;
out float v_size;

void main()
{
    int id = gl_VertexID;
    int col = id % 4;
    int row = id / 4;
    v_pos = vec2(-0.75 + 0.5 * float(col), -0.5 + 0.5 * float(row));
    v_color = vec3(0.25 + 0.046875 * float(id), 0.9375 - 0.046875 * float(id), 0.3125 * float(1 + id % 3));
    v_size = float(6 + 2 * (id % 3));
    gl_Position = vec4(v_pos, 0.0, 1.0);
    gl_PointSize = v_size;
}
