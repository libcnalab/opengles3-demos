#version 310 es
// Fragment shader of the "graphics" program. Eight cells of the picture each show one resource, so that
// the values the interface queries report can be checked by pixels as well:
//   cell 0  u_tint            explicit location 4, declared in BOTH shaders (REFERENCED_BY_VERTEX and _FRAGMENT)
//   cell 1  u_weights[3]      explicit locations 8, 9, 10 (consecutive array elements), fragment shader only
//   cell 2  u_light           struct: u_light.dir at location 20, u_light.power at 21 (consecutive members)
//   cell 3  u_tex             layout(binding = 2): never set with glUniform1i, texture bound to unit 2
//   cell 4  u_texarr[2]       layout(binding = 4, location = 30): elements use units 4 and 5
//   cell 5  Params            std140 block, binding 3, summed in the vertex shader (v_ubo)
//   cell 6  Cell cells[2]     array of blocks, binding 6 -> elements use bindings 6 and 7
//   cell 7  u_pick            no explicit location: set through glGetProgramResourceLocation
precision highp float;
precision highp int;

layout(location = 4) uniform vec4 u_tint;

layout(location = 8) uniform float u_weights[3];

struct Light
{
    vec3 dir;
    float power;
};
layout(location = 20) uniform Light u_light;

layout(binding = 2) uniform sampler2D u_tex;
layout(binding = 4, location = 30) uniform sampler2D u_texarr[2];

uniform ivec2 u_pick;

layout(std140, binding = 6) uniform Cell
{
    vec4 color;
    ivec2 id;
} cells[2];

in vec4 v_color;
in vec2 v_uv;
in vec3 v_ubo;
flat in int v_cell;

layout(location = 0) out vec4 o_color;
// An array of two fragment outputs at consecutive locations 1 and 2. The default framebuffer has only
// draw buffer 0, so these values are discarded; they exist to be enumerated by PROGRAM_OUTPUT.
layout(location = 1) out ivec2 o_id[2];

void main()
{
    vec3 c = vec3(0.0);
    switch (v_cell)
    {
    case 0: c = v_color.rgb * (u_tint.a + 0.5); break;   // vertex stage: a_color * tint.rgb
    case 1: c = vec3(u_weights[0], u_weights[1], u_weights[2]); break;
    case 2: c = u_light.dir * u_light.power; break;
    case 3: c = texture(u_tex, v_uv).rgb; break;
    case 4: c = (v_uv.x < 0.5 ? texture(u_texarr[0], v_uv) : texture(u_texarr[1], v_uv)).rgb; break;
    case 5: c = v_ubo; break;
    case 6: c = (v_uv.x < 0.5 ? cells[0].color : cells[1].color).rgb; break;
    case 7: c = vec3(float(u_pick.x) * 0.25, float(u_pick.y) * 0.25, 0.75); break;
    }
    o_color = vec4(c, 1.0);
    o_id[0] = cells[0].id;
    o_id[1] = cells[1].id + ivec2(v_cell);
}
