#version 310 es
// A FRAGMENT shader that reads two shader storage blocks: the accumulators the compute shader updated with atomic
// functions (Acc, as raw words) and the values the application computed on the CPU (Expect). One 16-pixel cell per
// accumulator: green where the words are equal, red where they are not. Cells 12-15 test nothing and stay dark.
precision highp float;
precision highp int;

layout(std430, binding = 0) readonly buffer Acc    { uint v[]; } acc;
layout(std430, binding = 1) readonly buffer Expect { uint v[]; } ex;

out vec4 o_color;

void main()
{
    int x = int(gl_FragCoord.x);
    int cell = x / 16;
    if (cell >= 12 || x % 16 == 0 || int(gl_FragCoord.y) % 16 == 0)
    {
        o_color = vec4(0.06, 0.07, 0.10, 1.0);
        return;
    }
    o_color = acc.v[cell] == ex.v[cell] ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
