#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §3.6 Comments (/* */ and //, not nested, "treated syntactically as a single space"), §3.2 (UTF-8 in comments mandated), §3.7 Tokens
// @feature: GLSL300.COMMENTS_TOKENS
precision mediump float;
out vec4 o;
/* a block comment
   spanning lines, containing // which does not start a comment here */
// a line comment containing /* which does not start a block comment
// UTF-8 inside a comment: Příliš žluťoučký kůň ∀x ∈ ℝ
void main()
{
    int/**/a/**/=/**/1/* a comment is a single space, so tokens on both sides stay separate */+2;
    float b = 0.5; // trailing comment
    o = vec4(float(a) * b);
}
