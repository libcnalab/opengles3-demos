#version 320 es
// The terrain is a grid of quad patches. A patch vertex is just a corner (x, z) in world space; everything else (LOD,
// height, projection) happens in the tessellation stages.
layout(location = 0) in vec2 a_xz;

void main()
{
    gl_Position = vec4(a_xz.x, 0.0, a_xz.y, 1.0);
}
