#version 310 es
// Instanced camera-facing billboard. Attribute 0 is the quad corner (per vertex); attributes 1 and 2 are read
// straight from the particle SSBO bound as ARRAY_BUFFER with a divisor of 1: one particle per instance.
precision highp float;

layout(location = 0) in vec2 a_corner;    // (-1,-1) .. (1,1)
layout(location = 1) in vec4 a_pos_age;   // Particle.pos, Particle.age  (bytes 0..15 of the std430 element)
layout(location = 2) in vec4 a_vel_life;  // Particle.vel, Particle.life (bytes 16..31)

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_half;      // half size in world units
uniform float u_min_px;    // minimum half size in pixels (keeps small windows readable)
uniform float u_px_scale;  // world size of one pixel at eye distance 1 = 2 tan(fovy / 2) / height
uniform float u_intensity;
uniform int u_flat;        // validation: constant colour, no falloff
uniform vec3 u_flat_color;

out vec2 v_corner;
out vec3 v_color;

void main()
{
    vec4 eye = u_view * vec4(a_pos_age.xyz, 1.0);
    float half_size = max(u_half, u_min_px * u_px_scale * -eye.z);
    eye.xy += a_corner * half_size;
    gl_Position = u_proj * eye;
    v_corner = a_corner;

    if (u_flat != 0)
    {
        v_color = u_flat_color;
        return;
    }
    // Colour: fast particles (just launched, or falling hard) are ice blue, slow ones near the apex turn warm;
    // the brightness fades in over the first steps and out towards the end of the life.
    float t = clamp(a_pos_age.w / a_vel_life.w, 0.0, 1.0);
    float speed = length(a_vel_life.xyz);
    vec3 slow = vec3(1.00, 0.42, 0.16);
    vec3 fast = vec3(0.30, 0.70, 1.00);
    vec3 c = mix(slow, fast, smoothstep(0.8, 5.0, speed));
    c = mix(c, vec3(0.75, 0.25, 0.95), smoothstep(0.55, 1.0, t));  // old particles drift to violet
    float fade = smoothstep(0.0, 0.05, t) * (1.0 - t);
    v_color = c * (u_intensity * fade);
}
