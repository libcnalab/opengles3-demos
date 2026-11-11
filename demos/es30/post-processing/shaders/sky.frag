#version 300 es
// Background of the 3D scene: a vertical gradient that follows the camera. The color depends on the
// elevation of the view ray of the pixel (not on the screen row), so the haze sits exactly on the horizon
// whatever the camera pitch is. The values are LINEAR light; the scene target stores them as they are
// (RGBA8) or encodes them on write (SRGB8_ALPHA8).
precision highp float;
precision highp int;

uniform vec2 u_resolution;                          // size of the render target in pixels
uniform vec3 u_cam_forward;                         // unit view direction
uniform vec3 u_cam_right;                           // right * tan(fovy / 2) * aspect
uniform vec3 u_cam_up;                              // up * tan(fovy / 2)
uniform vec3 u_horizon;                             // linear color at and below the horizon
uniform vec3 u_zenith;                              // linear color high in the sky

layout(location = 0) out vec4 o_color;

void main()
{
    vec2 ndc = 2.0 * gl_FragCoord.xy / u_resolution - 1.0;
    vec3 dir = normalize(u_cam_forward + u_cam_right * ndc.x + u_cam_up * ndc.y);
    float s = smoothstep(0.0, 0.25, dir.y);
    o_color = vec4(mix(u_horizon, u_zenith, s), 1.0);
}
