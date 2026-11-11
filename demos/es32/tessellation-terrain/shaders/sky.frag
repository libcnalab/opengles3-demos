#version 320 es
// Sky gradient; the application inserts terrain_common.glsl (sky_color, shared with the fog colour) after the #version line.
uniform vec2 u_viewport;   // window size in pixels

out vec4 o_color;

void main()
{
    o_color = vec4(sky_color(gl_FragCoord.y / u_viewport.y), 1.0);
}
