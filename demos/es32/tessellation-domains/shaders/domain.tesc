#version 320 es
// Tessellation control shader (ES 3.2 section 11.2.1, GLSL ES 3.20 section 4.4.2 "Tessellation Control Outputs").
//
// OUT_VERTICES is #defined by the application right after the #version line: the size of the OUTPUT patch. The shader is
// invoked once per output vertex; gl_InvocationID says which one. The input patch (gl_in, gl_PatchVerticesIn vertices) may
// have a different size: PATCH_VERTICES is application state, TESS_CONTROL_OUTPUT_VERTICES is a property of this shader.
layout(vertices = OUT_VERTICES) out;

uniform vec4 u_outer;   // requested gl_TessLevelOuter
uniform vec2 u_inner;   // requested gl_TessLevelInner
uniform float u_bias;   // added to every level, times gl_PrimitiveID (0 for single patches)

in float v_tag[];       // from the vertex shader, one value per input vertex

out float t_tag[];      // per-vertex outputs: arrays indexed by gl_InvocationID (written only at that index)
out vec3 t_pos[];
patch out vec4 p_center;  // per-patch output (not indexed): centroid of the output patch (xyz), input patch size (w)

void main()
{
    // Each invocation passes on its own vertex.
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    t_pos[gl_InvocationID] = gl_in[gl_InvocationID].gl_Position.xyz;
    t_tag[gl_InvocationID] = v_tag[gl_InvocationID];

    // Invocation 0 is about to read what the OTHER invocations wrote: without this barrier() the result is undefined
    // (GLSL ES 3.20 section 8.15). barrier() must be called unconditionally in main().
    barrier();

    if (gl_InvocationID == 0)
    {
        vec3 sum = vec3(0.0);
        for (int i = 0; i < OUT_VERTICES; ++i) sum += t_pos[i];
        p_center = vec4(sum / float(OUT_VERTICES), float(gl_PatchVerticesIn));

        float bias = u_bias * float(gl_PrimitiveID);  // gl_PrimitiveID numbers the patches of the draw call, from zero
        gl_TessLevelOuter[0] = u_outer.x + bias;
        gl_TessLevelOuter[1] = u_outer.y + bias;
        gl_TessLevelOuter[2] = u_outer.z + bias;
        gl_TessLevelOuter[3] = u_outer.w + bias;
        gl_TessLevelInner[0] = u_inner.x + bias;
        gl_TessLevelInner[1] = u_inner.y + bias;
    }
}
