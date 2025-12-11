// Image shader for WebGPU rendering
// Renders textured quads for image glyphs

// Uniform buffer containing rendering parameters
struct Uniforms {
    canvas_size: vec2f,
    global_alpha: f32,
    _pad0: f32,
    // Image bounds in screen coordinates: (x0, y0, x1, y1)
    bounds: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var image_texture: texture_2d<f32>;
@group(0) @binding(2) var image_sampler: sampler;

// Vertex input from quad geometry (per-vertex)
struct VertexInput {
    @location(0) position: vec2f,  // -0.5 to 0.5
}

// Data passed from vertex to fragment shader
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) tex_coords: vec2f,
}

@vertex
fn vertex_main(vertex: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Map -0.5..0.5 to 0..1 for texture coordinates
    output.tex_coords = vec2f(
        select(0.0, 1.0, vertex.position.x > 0.0),
        select(0.0, 1.0, vertex.position.y > 0.0)
    );

    // Compute screen position from bounds
    // bounds: (x0, y0, x1, y1)
    let x = select(uniforms.bounds.x, uniforms.bounds.z, vertex.position.x > 0.0);
    let y = select(uniforms.bounds.y, uniforms.bounds.w, vertex.position.y > 0.0);

    // Convert to normalized device coordinates
    var pos = vec2f(x, y) + 0.5;  // Bokeh's offset
    pos = pos / uniforms.canvas_size;

    output.position = vec4f(2.0 * pos.x - 1.0, 1.0 - 2.0 * pos.y, 0.0, 1.0);

    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
    let color = textureSample(image_texture, image_sampler, input.tex_coords);
    let alpha = color.a * uniforms.global_alpha;
    // Premultiplied alpha
    return vec4f(color.rgb * alpha, alpha);
}
