#version 310 es
// The "uniform zoo": a separable FRAGMENT program whose default-block uniforms cover every type that
// glProgramUniform* can load (float/int/uint scalars and vectors, their array forms, all nine matrix
// shapes, bool vectors, structs, partial arrays). The application sets ALL of them with
// glProgramUniform* while the program is neither current (glUseProgram is never called) nor the
// active program of the pipeline. Each cell of an 8-column grid checks one uniform against the exact
// value that was uploaded: green = the shader sees exactly that value, red = it does not. All values
// are exactly representable; highp is used for float and int (a mediump int may be only 16 bits, too
// few for 0x80000001u).
precision highp float;
precision highp int;

uniform ivec2 u_cell_px; // cell size in pixels (width, height): the grid fills the window
uniform ivec2 u_origin;  // window position of the grid's lower-left corner

uniform float u_f;      uniform vec2 u_v2;      uniform vec3 u_v3;      uniform vec4 u_v4;
uniform float u_farr[2]; uniform vec2 u_v2arr[2]; uniform vec3 u_v3arr[2]; uniform vec4 u_v4arr[2];
uniform int u_i;        uniform ivec2 u_iv2;    uniform ivec3 u_iv3;    uniform ivec4 u_iv4;
uniform int u_iarr[2];  uniform ivec2 u_iv2arr[2]; uniform ivec3 u_iv3arr[2]; uniform ivec4 u_iv4arr[2];
uniform uint u_u;       uniform uvec2 u_uv2;    uniform uvec3 u_uv3;    uniform uvec4 u_uv4;
uniform uint u_uarr[2]; uniform uvec2 u_uv2arr[2]; uniform uvec3 u_uv3arr[2]; uniform uvec4 u_uv4arr[2];
uniform mat2 u_m2;      uniform mat3 u_m3;      uniform mat4 u_m4;
uniform mat2x3 u_m23;   uniform mat3x2 u_m32;   uniform mat2x4 u_m24;
uniform mat4x2 u_m42;   uniform mat3x4 u_m34;   uniform mat4x3 u_m43;
uniform bvec2 u_bv2;
struct S { float a; ivec2 b; };
uniform S u_s;
struct T { vec3 c; };
uniform T u_sa[2];
uniform float u_part[4];
uniform mat2 u_marr[2];

layout(location = 0) out vec4 o_color;

bool check(int i)
{
    switch (i)
    {
    case 0:  return u_f == 0.5;
    case 1:  return u_v2 == vec2(0.25, -1.5);
    case 2:  return u_v3 == vec3(1.0, 2.0, -3.0);
    case 3:  return u_v4 == vec4(0.125, 0.375, -0.625, 8.0);
    case 4:  return u_farr[0] == 1.5 && u_farr[1] == -2.5;
    case 5:  return u_v2arr[0] == vec2(1.0, 2.0) && u_v2arr[1] == vec2(3.0, 4.0);
    case 6:  return u_v3arr[0] == vec3(1.0, 2.0, 3.0) && u_v3arr[1] == vec3(4.0, 5.0, 6.0);
    case 7:  return u_v4arr[0] == vec4(1.0, 2.0, 3.0, 4.0) && u_v4arr[1] == vec4(5.0, 6.0, 7.0, 8.0);
    case 8:  return u_i == -7;
    case 9:  return u_iv2 == ivec2(1, -2);
    case 10: return u_iv3 == ivec3(3, -4, 5);
    case 11: return u_iv4 == ivec4(-6, 7, -8, 9);
    case 12: return u_iarr[0] == 100 && u_iarr[1] == -100;
    case 13: return u_iv2arr[0] == ivec2(1, 2) && u_iv2arr[1] == ivec2(3, 4);
    case 14: return u_iv3arr[0] == ivec3(1, 2, 3) && u_iv3arr[1] == ivec3(4, 5, 6);
    case 15: return u_iv4arr[0] == ivec4(1, 2, 3, 4) && u_iv4arr[1] == ivec4(5, 6, 7, 8);
    case 16: return u_u == 0x80000001u;
    case 17: return u_uv2 == uvec2(1u, 0xFFFFFFFFu);
    case 18: return u_uv3 == uvec3(10u, 20u, 30u);
    case 19: return u_uv4 == uvec4(1u, 2u, 3u, 4u);
    case 20: return u_uarr[0] == 7u && u_uarr[1] == 9u;
    case 21: return u_uv2arr[0] == uvec2(1u, 2u) && u_uv2arr[1] == uvec2(3u, 4u);
    case 22: return u_uv3arr[0] == uvec3(1u, 2u, 3u) && u_uv3arr[1] == uvec3(4u, 5u, 6u);
    case 23: return u_uv4arr[0] == uvec4(1u, 2u, 3u, 4u) && u_uv4arr[1] == uvec4(5u, 6u, 7u, 8u);
    // Matrices are column-major: matCxR(a, b, ...) fills column 0 first. m[c][r] = column c, row r.
    case 24: return u_m2 == mat2(1.0, 2.0, 3.0, 4.0) && u_m2[1][0] == 3.0;
    case 25: return u_m3 == mat3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) && u_m3[2][1] == 8.0;
    case 26: return u_m4 == mat4(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0) && u_m4[3][0] == 13.0;
    case 27: return u_m23 == mat2x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0) && u_m23[1][2] == 6.0;   // 2 columns x 3 rows
    case 28: return u_m32 == mat3x2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0) && u_m32[2][1] == 6.0;   // 3 columns x 2 rows
    case 29: return u_m24 == mat2x4(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) && u_m24[1][3] == 8.0;
    case 30: return u_m42 == mat4x2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) && u_m42[3][1] == 8.0;
    case 31: return u_m34 == mat3x4(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0) && u_m34[2][3] == 12.0;
    case 32: return u_m43 == mat4x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0) && u_m43[3][2] == 12.0;
    case 33: return u_bv2 == bvec2(true, false);
    case 34: return u_s.a == 0.75 && u_s.b == ivec2(4, -4);
    case 35: return u_sa[0].c == vec3(1.0) && u_sa[1].c == vec3(9.0, 8.0, 7.0);
    case 36: return u_part[0] == 0.0 && u_part[1] == 1.0 && u_part[2] == 42.0 && u_part[3] == 3.0;
    case 37: return u_marr[0] == mat2(1.0, 2.0, 3.0, 4.0) && u_marr[1] == mat2(5.0, 6.0, 7.0, 8.0);
    }
    return false;
}

void main()
{
    ivec2 cell = (ivec2(gl_FragCoord.xy) - u_origin) / u_cell_px;
    int index = cell.y * 8 + cell.x;
    if (index >= 38) { o_color = vec4(0.1, 0.1, 0.12, 1.0); return; }
    o_color = check(index) ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
