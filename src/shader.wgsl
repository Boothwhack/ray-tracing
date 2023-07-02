struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vertex_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = vec4(in.position, 1.0);
    out.tex_coords = in.tex_coords;

    return out;
}

@group(0) @binding(0)
var frame_texture: texture_2d<f32>;
@group(0) @binding(1)
var frame_sampler: sampler;

struct FragmentInput {
    @location(0) tex_coords: vec2<f32>,
}

@fragment
fn fragment_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return textureSample(frame_texture, frame_sampler, in.tex_coords);
}
