#version 310 es
// A vertex shader with no inputs and no outputs: one triangle that covers the viewport. Paired with the
// small fragment shaders below, which exist to test explicit uniform locations.
void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));  // (0,0) (2,0) (0,2)
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
