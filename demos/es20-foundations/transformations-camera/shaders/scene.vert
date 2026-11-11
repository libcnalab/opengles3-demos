#version 300 es
// The whole coordinate chain of the demo is in this shader, one line per stage.
//
//   a_position            object (model) space   the vertex as stored in the buffer
//   u_model * a_position  world space            places the object in the scene
//   u_view  * world       eye (view) space       the camera sits at the origin and looks down -z
//   u_projection * eye    clip space             gl_Position; the clip volume is -w <= x,y,z <= w
//
// After this shader the fixed-function part of the pipeline takes over: clipping against the clip
// volume, the perspective divide (clip.xyz / clip.w = normalized device coordinates in [-1, 1]) and the
// viewport transformation (glViewport / glDepthRangef) that gives window coordinates.
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_color;

// Three separate matrices on purpose: each upload is one stage of the chain. A real renderer would
// usually premultiply them on the CPU into one MVP.
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

// A fixed directional light in WORLD space, only there so that faces with different orientations get
// different brightness and the cube reads as a solid. (Proper lighting is a later demo.)
uniform vec3 u_light_dir;   // unit vector pointing towards the light
uniform float u_ambient;
uniform float u_lit;        // 1.0 = shaded, 0.0 = flat color (the axis gizmo)

out vec3 v_color;

void main()
{
    vec4 world = u_model * vec4(a_position, 1.0);
    vec4 eye = u_view * world;
    gl_Position = u_projection * eye;

    // mat3(u_model) is the right transform for normals only because every model matrix of this demo is
    // a rotation, a translation and a uniform scale; normalize() removes the scale.
    vec3 n = normalize(mat3(u_model) * a_normal);
    float diffuse = max(dot(n, u_light_dir), 0.0);
    float shade = mix(1.0, u_ambient + (1.0 - u_ambient) * diffuse, u_lit);
    v_color = a_color * shade;
}
