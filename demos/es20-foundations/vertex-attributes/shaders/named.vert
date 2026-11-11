#version 300 es
// Deliberately WITHOUT layout(location): the generic attribute index of every input is chosen
// either by the application (glBindAttribLocation before glLinkProgram) or by the linker
// (read back with glGetAttribLocation). ES 3.0 2.12.5.
in vec2 a_position;
in vec3 a_color;
in vec3 a_unused;   // declared but never read: not an *active* attribute, so it has no location

out vec4 v_color;

void main()
{
    v_color = vec4(a_color, 1.0);
    gl_Position = vec4(a_position, 0.0, 1.0);
}
