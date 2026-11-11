#version 300 es
// Visualizes one scene depth texture at its own resolution (the viewport is the texture size).
// The stored window depth d is turned back into an eye-space position, its screen-space
// derivatives give a surface normal ("relief" shading), and lines of constant eye distance are
// drawn every world unit. Coarse depth quantization (DEPTH_COMPONENT16 far away) shows up as
// stair-stepped normals and jagged contour lines; 24-bit and 32F stay smooth.
precision highp float;
precision highp sampler2D;

uniform sampler2D u_depth;   // compare mode NONE through a NEAREST sampler object: r = Dt
uniform vec2 u_near_far;     // projection near / far of the camera that rendered the texture
uniform vec2 u_tan;          // tan(fovy / 2) * aspect, tan(fovy / 2)
uniform vec2 u_size;         // texture size in texels
uniform vec3 u_sky;          // color for d == 1.0 (cleared depth: nothing was drawn)

out vec4 o_color;

float eye_distance(float d)
{
    float n = u_near_far.x;
    float f = u_near_far.y;
    float z_ndc = d * 2.0 - 1.0;
    return 2.0 * n * f / ((f + n) - z_ndc * (f - n));
}

vec3 palette(float t)
{
    // near: warm amber -> rose -> violet -> deep teal far away.
    vec3 c0 = vec3(1.00, 0.74, 0.34);
    vec3 c1 = vec3(0.93, 0.38, 0.47);
    vec3 c2 = vec3(0.46, 0.30, 0.72);
    vec3 c3 = vec3(0.12, 0.40, 0.52);
    t = clamp(t, 0.0, 1.0) * 3.0;
    if (t < 1.0) return mix(c0, c1, t);
    if (t < 2.0) return mix(c1, c2, t - 1.0);
    return mix(c2, c3, t - 2.0);
}

void main()
{
    float d = texelFetch(u_depth, ivec2(gl_FragCoord.xy), 0).r;
    float z = eye_distance(d);
    vec2 ndc = gl_FragCoord.xy / u_size * 2.0 - 1.0;
    vec3 p = vec3(ndc * u_tan * z, -z);
    vec3 n = normalize(cross(dFdx(p), dFdy(p)));
    vec3 l = normalize(vec3(-0.45, 0.80, 0.40));
    float diffuse = clamp(dot(n, l), 0.0, 1.0);

    float t = log(z / 2.0) / log(60.0 / 2.0);
    vec3 base = palette(t);

    float fw = fwidth(z);
    float dist = abs(fract(z + 0.5) - 0.5);                 // distance to the nearest whole unit
    float line = (1.0 - smoothstep(0.0, 1.2 * fw, dist)) * (1.0 - smoothstep(0.25, 0.6, fw));

    vec3 color = base * (0.32 + 0.68 * diffuse) * (1.0 - 0.45 * line);
    if (d >= 1.0)
        color = u_sky;
    o_color = vec4(color, 1.0);
}
