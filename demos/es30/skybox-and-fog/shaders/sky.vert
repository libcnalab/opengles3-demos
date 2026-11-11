#version 300 es
// Skybox with the depth trick.
//
// The sky is a unit cube centred on the camera. Only the ROTATION of the view matrix is applied
// (mat3(u_view) drops the translation), so the cube moves with the camera and the sky stays
// infinitely far away. The object-space corner position doubles as the cube-map direction.
//
// gl_Position = clip.xyww sets z = w, so after the perspective division z_ndc = w / w = 1 and the
// viewport transformation gives window depth 1.0 (the far plane) for every sky fragment. With the
// depth buffer cleared to 1.0 and glDepthFunc(GL_LEQUAL) the sky passes only where nothing nearer
// was drawn. u_depth_trick = false keeps the cube at its true depth (about 0.5 units from the
// eye): it then hides the whole scene. validate() uses that as a negative control.
layout(location = 0) in vec3 a_position;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform bool u_depth_trick;

out vec3 v_dir;

void main()
{
    v_dir = a_position;
    vec4 clip = u_proj * vec4(mat3(u_view) * a_position, 1.0);
    gl_Position = u_depth_trick ? clip.xyww : clip;
}
