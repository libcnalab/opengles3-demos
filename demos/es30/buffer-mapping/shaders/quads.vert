#version 300 es
// Draws vertex data that the application wrote into a buffer object through a mapped pointer.
// Every vertex is { vec2 position (floats), vec4 color (four normalized unsigned bytes) }, 12 bytes.
// The shader does nothing but forward them, so the picture is exactly the buffer contents.
//
// u_shift moves a whole draw in NDC. It lets one buffer hold a "template" quad that is drawn once in
// place and once shifted to another slot of the cell - used to show the result of glCopyBufferSubData.
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec4 a_color;

uniform vec2 u_shift;

out vec4 v_color;

void main()
{
    gl_Position = vec4(a_position + u_shift, 0.0, 1.0);
    v_color = a_color;
}
