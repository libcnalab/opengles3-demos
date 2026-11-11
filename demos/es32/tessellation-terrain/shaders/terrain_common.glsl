// Shared by the tessellation control, tessellation evaluation and fragment shaders. The application inserts this text
// right after the #version line, together with `#define HEIGHTMAP_N <texels per edge>` and `#define TERRAIN_HALF <half edge, world units>`
// (GLSL has no #include; keeping one copy guarantees that the three stages see the same terrain).
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_heightmap;   // R16F, HEIGHTMAP_N x HEIGHTMAP_N: one texel per lattice point of the terrain

const vec3 kHorizon = vec3(0.72, 0.82, 0.93);
const vec3 kZenith = vec3(0.18, 0.38, 0.72);

// Sky colour as a function of the height in the window (0 = bottom row, 1 = top row).
vec3 sky_color(float y01)
{
    float t = clamp(y01 * 1.25 - 0.1, 0.0, 1.0);
    return mix(kHorizon, kZenith, t * t);
}

float lattice_height(ivec2 i)
{
    return texelFetch(u_heightmap, clamp(i, ivec2(0), ivec2(HEIGHTMAP_N - 1)), 0).r;
}

// Height of the terrain at a world position (x, z): bilinear interpolation of the four lattice heights around it. It is
// written out with texelFetch instead of texture(): the result then does not depend on the implementation's
// filtering precision, and a CPU evaluation of the same formula gives the same number.
float height_at(vec2 xz)
{
    float last = float(HEIGHTMAP_N - 1);
    vec2 f = clamp((xz + vec2(TERRAIN_HALF)) * (last / (2.0 * TERRAIN_HALF)), vec2(0.0), vec2(last - 0.001));
    ivec2 i = ivec2(floor(f));
    vec2 w = f - vec2(i);
    float h0 = mix(lattice_height(i), lattice_height(i + ivec2(1, 0)), w.x);
    float h1 = mix(lattice_height(i + ivec2(0, 1)), lattice_height(i + ivec2(1, 1)), w.x);
    return mix(h0, h1, w.y);
}

// Surface normal from central differences of the height field, one lattice spacing apart.
vec3 terrain_normal(vec2 xz)
{
    float d = 2.0 * TERRAIN_HALF / float(HEIGHTMAP_N - 1);
    float hx0 = height_at(xz - vec2(d, 0.0));
    float hx1 = height_at(xz + vec2(d, 0.0));
    float hz0 = height_at(xz - vec2(0.0, d));
    float hz1 = height_at(xz + vec2(0.0, d));
    return normalize(vec3(hx0 - hx1, 2.0 * d, hz0 - hz1));
}

// Lighting model: material by height and slope (lake, sand, grass, rock, snow), a sun and an ambient term, linear fog to the
// colour of the horizon. Only +, *, mix, smoothstep, normalize and length are used: no transcendental functions, so a
// CPU evaluation matches the GPU closely.
vec3 terrain_shade(vec3 world, vec3 eye)
{
    vec2 xz = world.xz;
    float h = height_at(xz);
    vec3 n = terrain_normal(xz);
    float water = 1.0 - smoothstep(0.62, 0.95, h);     // the height map is clamped at 0.6: the flat parts are lakes
    n = normalize(mix(n, vec3(0.0, 1.0, 0.0), water));
    float slope = 1.0 - n.y;
    vec3 albedo = mix(vec3(0.74, 0.68, 0.48), vec3(0.22, 0.40, 0.13), smoothstep(0.7, 1.6, h));
    albedo = mix(albedo, vec3(0.45, 0.40, 0.35), smoothstep(0.12, 0.30, slope));
    albedo = mix(albedo, vec3(0.96, 0.97, 1.00), smoothstep(4.6, 5.8, h) * (1.0 - smoothstep(0.30, 0.55, slope)));
    albedo = mix(albedo, vec3(0.08, 0.28, 0.42), water);
    vec3 sun = normalize(vec3(0.60, 0.50, 0.35));
    float diffuse = max(dot(n, sun), 0.0);
    vec3 lit = albedo * (0.26 + 0.74 * diffuse);
    float fog = clamp((length(world - eye) - 30.0) / 90.0, 0.0, 1.0);
    return mix(lit, kHorizon, fog);
}
