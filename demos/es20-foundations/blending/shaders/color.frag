#version 300 es
// The fragment shader only produces the *source color* of the blend equation (Rs, Gs, Bs, As).
// Everything that happens afterwards - factors, equation, clamping - is fixed-function state
// set with glBlendFunc*, glBlendEquation* and glBlendColor (ES 3.0 section 4.1.7).
precision highp float;

uniform vec4 u_color;  // source color, exactly representable in 8 bits (k / 255)
uniform int u_disc;    // 0: fill the rectangle, 1: keep only the inscribed disc

in vec2 v_local;

// One user-declared output replaces gl_FragColor; location 0 is draw buffer 0.
layout(location = 0) out vec4 o_color;

void main()
{
    // discard ends the invocation: no fragment reaches blending, so the framebuffer is untouched.
    if (u_disc != 0 && dot(v_local, v_local) > 1.0)
        discard;
    o_color = u_color;
}
