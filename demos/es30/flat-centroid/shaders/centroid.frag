#version 300 es
// Shows the atlas tile, or - in the measuring modes - discards every fragment that does NOT have the
// property under test, so an ANY_SAMPLES_PASSED occlusion query answers "did any fragment have it?"
// without depending on the implementation-dependent multisample resolve.
//   u_mode 0  display
//   u_mode 1  keep fragments whose edge distance is < -u_epsilon  (evaluated outside the primitive)
//   u_mode 2  keep fragments whose texel belongs to another atlas cell (texture bleed)
//   u_mode 3  positive control: keep fragments evaluated inside the tile, less than 2 px from an edge
//   u_mode 4  keep fragments whose centroid value differs from the non-centroid value
precision highp float;

uniform sampler2D u_atlas;
uniform bool u_use_centroid;
uniform int u_mode;
uniform float u_epsilon;     // pixels
uniform float u_cell_alpha;  // alpha tag of the cell the tile is meant to show

in vec2 v_uv;
centroid in vec2 v_uv_centroid;
in vec4 v_dist;
centroid in vec4 v_dist_centroid;

layout(location = 0) out vec4 o_color;

void main()
{
    vec2 uv = u_use_centroid ? v_uv_centroid : v_uv;
    vec4 d = u_use_centroid ? v_dist_centroid : v_dist;
    vec4 texel = texture(u_atlas, uv);

    float nearest_edge = min(min(d.x, d.y), min(d.z, d.w));
    bool outside = nearest_edge < -u_epsilon;
    bool bleed = abs(texel.a - u_cell_alpha) > 0.5 / 255.0;
    bool differs = any(greaterThan(abs(v_dist_centroid - v_dist), vec4(1.0e-3)));

    if (u_mode == 1 && !outside) discard;
    if (u_mode == 2 && !bleed) discard;
    if (u_mode == 3 && !(nearest_edge >= -u_epsilon && nearest_edge < 2.0)) discard;
    if (u_mode == 4 && !differs) discard;

    o_color = vec4(texel.rgb, 1.0);
}
