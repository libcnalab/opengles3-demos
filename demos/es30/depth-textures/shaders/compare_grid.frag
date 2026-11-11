#version 300 es
// One cell of the comparison-function grid. The same DEPTH_COMPONENT32F texture is bound twice:
// once through a NEAREST sampler object with compare mode NONE (raw depth Dt), once through a
// sampler object with COMPARE_REF_TO_TEXTURE and the cell's compare function.
//
//   upper part : Dref is a "slicing surface" at an eye distance that grows from left to right;
//                the cell shows where the function passes against the scene depth.
//   bottom band: Dref is the raw depth of the very same texel, so Dref == Dt exactly and the
//                band is lit for LEQUAL, GEQUAL, EQUAL, ALWAYS and dark for the other four
//                (ES 3.0 Table 3.23).
precision highp float;
precision highp sampler2D;
precision highp sampler2DArray;
precision highp samplerCube;
precision highp sampler2DShadow;
precision highp sampler2DArrayShadow;
precision highp samplerCubeShadow;

uniform sampler2D u_raw2d;                    // unit 0
uniform sampler2DShadow u_shadow2d;           // unit 1
uniform sampler2DArrayShadow u_shadow_array;  // unit 2
uniform samplerCubeShadow u_shadow_cube;      // unit 3
uniform sampler2DArray u_raw_array;           // unit 4
uniform samplerCube u_raw_cube;               // unit 5

uniform int u_kind;          // 0 = sampler2DShadow, 1 = sampler2DArrayShadow (layer 1), 2 = samplerCubeShadow
uniform vec4 u_cell;         // x0, y0, width, height in window pixels
uniform float u_band;        // height fraction of the exact band
uniform vec3 u_pass;         // color of the function where it passes
uniform vec2 u_near_far;     // projection of the texture
uniform vec2 u_slice;        // eye distance of the slicing surface at the left / right cell edge

out vec4 o_color;

float window_depth(float z)  // eye distance -> window depth with DepthRangef(0, 1)
{
    float n = u_near_far.x;
    float f = u_near_far.y;
    return 0.5 + 0.5 * ((f + n) / (f - n) - 2.0 * f * n / ((f - n) * z));
}

void main()
{
    vec2 local = (gl_FragCoord.xy - u_cell.xy) / u_cell.zw;
    bool band = local.y < u_band;
    float slice = mix(u_slice.x, u_slice.y, local.x);

    float raw;
    float r;
    if (u_kind == 0)
    {
        raw = texture(u_raw2d, local).r;
        float ref = band ? raw : window_depth(slice);
        r = texture(u_shadow2d, vec3(local, ref));
    }
    else if (u_kind == 1)
    {
        raw = texture(u_raw_array, vec3(local, 1.0)).r;
        float ref = band ? raw : window_depth(slice);
        r = texture(u_shadow_array, vec4(local, 1.0, ref));
    }
    else
    {
        // Equirectangular view around the cube center.
        float lon = (local.x * 2.0 - 1.0) * 3.14159265;
        float lat = (local.y - 0.5) * 3.14159265;
        vec3 dir = vec3(cos(lat) * sin(lon), sin(lat), -cos(lat) * cos(lon));
        raw = texture(u_raw_cube, dir).r;
        // A sphere of radius `slice`: along dir its face-space z is slice * max(|x|, |y|, |z|).
        vec3 a = abs(dir);
        float ref = band ? raw : window_depth(slice * max(a.x, max(a.y, a.z)));
        r = texture(u_shadow_cube, vec4(dir, ref));
    }

    float n = u_near_far.x;
    float f = u_near_far.y;
    float z = 2.0 * n * f / ((f + n) - (raw * 2.0 - 1.0) * (f - n));
    float shade = raw >= 1.0 ? 0.0 : clamp(1.0 - log(z / 1.5) / log(40.0), 0.0, 1.0);

    vec3 color;
    if (band)
        color = r > 0.5 ? u_pass : u_pass * 0.10;
    else
        color = r > 0.5 ? u_pass * (0.70 + 0.30 * shade) : u_pass * (0.06 + 0.14 * shade);
    o_color = vec4(color, 1.0);
}
