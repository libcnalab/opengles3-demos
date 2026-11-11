#version 300 es
// Writes a ramp of LINEAR values, one per pixel column of a 256-pixel-wide target:
//     l = x / 255      (x = 0 .. 255)
//     color = (l, 1 - l, l * l, 1 - l)
// If the target's FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING is SRGB, the GL encodes red, green and blue
// to the sRGB curve on the way into the 8-bit target (ES 3.0 4.1.8); alpha is never encoded.
precision highp float;

out vec4 o_color;

void main()
{
    float l = float(int(gl_FragCoord.x)) / 255.0;
    o_color = vec4(l, 1.0 - l, l * l, 1.0 - l);
}
