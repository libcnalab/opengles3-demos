#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §7.3 Built-In Constants are const ints usable in constant expressions; each is at least the listed minimum
// @feature: GLSL300.BUILTIN_CONSTANTS
precision mediump float;
out vec4 o;
float fv[gl_MaxFragmentUniformVectors];
void main()
{
    float a[(gl_MaxVertexAttribs >= 16 && gl_MaxVertexUniformVectors >= 256 && gl_MaxVertexOutputVectors >= 16
             && gl_MaxFragmentInputVectors >= 15 && gl_MaxVertexTextureImageUnits >= 16
             && gl_MaxCombinedTextureImageUnits >= 32 && gl_MaxTextureImageUnits >= 16
             && gl_MaxFragmentUniformVectors >= 224 && gl_MaxDrawBuffers >= 4
             && gl_MinProgramTexelOffset <= -8 && gl_MaxProgramTexelOffset >= 7) ? 1 : -1];
    o = vec4(float(a.length() + fv.length()));
}
