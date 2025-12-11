// Accumulate shader for blitting WebGPU framebuffer to canvas
// This shader renders a full-screen quad with the accumulated texture

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) tex_coord: vec2f,
}

// Full-screen quad vertices (triangle strip)
var<private> positions: array<vec2f, 4> = array<vec2f, 4>(
    vec2f(-1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
);

var<private> tex_coords: array<vec2f, 4> = array<vec2f, 4>(
    vec2f(0.0, 1.0),
    vec2f(0.0, 0.0),
    vec2f(1.0, 1.0),
    vec2f(1.0, 0.0),
);

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4f(positions[vertex_index], 0.0, 1.0);
    output.tex_coord = tex_coords[vertex_index];
    return output;
}

@group(0) @binding(0) var framebuffer_texture: texture_2d<f32>;
@group(0) @binding(1) var framebuffer_sampler: sampler;

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
    return textureSample(framebuffer_texture, framebuffer_sampler, input.tex_coord);
}
