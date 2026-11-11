#version 300 es
// One step of a particle system. The vertex shader is used as a tiny "compute" stage: it runs once per
// particle and its outputs are recorded by transform feedback. All arithmetic uses values that are exact
// in a float (small dyadic fractions), so the CPU model reproduces every bit.
//
// The same shader is linked into one program with the varyings v_pos, v_vel, v_color, v_size captured in
// INTERLEAVED_ATTRIBS mode: 8 floats = 32 bytes per particle, the same layout as the input arrays, so a
// captured buffer can be fed straight back in as the next step's vertex data (ping-pong).
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_vel;
layout(location = 2) in vec3 a_color;
layout(location = 3) in float a_size;

uniform float u_dt;
uniform vec2 u_gravity;
uniform float u_floor;

out vec2 v_pos;
out vec2 v_vel;
out vec3 v_color;
out float v_size;

void main()
{
    vec2 vel = a_vel + u_gravity * u_dt;
    vec2 pos = a_pos + vel * u_dt;
    if (pos.y < u_floor)
    {
        pos.y = 2.0 * u_floor - pos.y;  // mirror at the floor and lose half of the vertical speed
        vel.y = -0.5 * vel.y;
    }
    v_pos = pos;
    v_vel = vel;
    v_color = a_color;
    v_size = a_size;
    gl_Position = vec4(pos, 0.0, 1.0);
    gl_PointSize = a_size;  // only visible when RASTERIZER_DISCARD is off
}
