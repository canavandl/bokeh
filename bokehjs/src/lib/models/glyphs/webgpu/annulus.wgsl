// Annulus shader for WebGPU rendering
// Implements a ring/donut shape (circle with a hole)

// Constants
const PI: f32 = 3.14159265358979323846;

// Cap type enum values
const CAP_BUTT: i32 = 0;
const CAP_ROUND: i32 = 1;
const CAP_SQUARE: i32 = 2;

// Join type enum values
const JOIN_MITER: i32 = 0;
const JOIN_ROUND: i32 = 1;
const JOIN_BEVEL: i32 = 2;

// Uniform buffer containing rendering parameters
struct Uniforms {
    canvas_size: vec2f,
    antialias: f32,
    _pad0: f32,
    // Coordinate transform: screen = data * scale + offset
    x_scale: f32,
    x_offset: f32,
    y_scale: f32,
    y_offset: f32,
    // Radius transform (for data-unit radius)
    radius_scale: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Vertex input from quad geometry (per-vertex)
struct VertexInput {
    @location(0) position: vec2f,
}

// Instance data buffers
struct PositionInput {
    @location(1) center: vec2f,
}

struct RadiiInput {
    @location(2) radii: vec2f,  // inner_radius, outer_radius
}

struct LinePropsInput {
    @location(3) line_props: vec4f,  // linewidth, cap, join, show
}

struct LineColorInput {
    @location(4) line_color: vec4f,
}

struct FillColorInput {
    @location(5) fill_color: vec4f,
}

// Data passed from vertex to fragment shader
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) v_coords: vec2f,
    @location(1) v_inner_radius: f32,
    @location(2) v_outer_radius: f32,
    @location(3) v_linewidth: f32,
    @location(4) v_line_color: vec4f,
    @location(5) v_fill_color: vec4f,
    @location(6) v_line_cap: f32,
    @location(7) v_line_join: f32,
}

// Calculate enclosing size for the annulus including line width and antialiasing
fn enclosing_size(outer_radius: f32, linewidth: f32) -> f32 {
    return outer_radius + linewidth + uniforms.antialias;
}

@vertex
fn vertex_main(
    vertex: VertexInput,
    pos_in: PositionInput,
    radii_in: RadiiInput,
    line_props_in: LinePropsInput,
    line_color_in: LineColorInput,
    fill_color_in: FillColorInput
) -> VertexOutput {
    var output: VertexOutput;

    // Extract line properties
    let linewidth_raw = line_props_in.line_props.x;
    let line_cap = line_props_in.line_props.y;
    let line_join = line_props_in.line_props.z;
    let show = line_props_in.line_props.w;

    let inner_radius = radii_in.radii.x;
    let outer_radius = radii_in.radii.y;

    // Early exit for hidden markers
    if (show < 0.5 || outer_radius < 0.0) {
        output.position = vec4f(-2.0, -2.0, 0.0, 1.0);
        return output;
    }

    // Transform data coordinates to screen coordinates
    let screen_center = vec2f(
        pos_in.center.x * uniforms.x_scale + uniforms.x_offset,
        pos_in.center.y * uniforms.y_scale + uniforms.y_offset
    );

    // Transform radii from data units to screen units
    let screen_inner_radius = inner_radius * uniforms.radius_scale;
    let screen_outer_radius = outer_radius * uniforms.radius_scale;

    // Adjust line color alpha for thin lines
    var v_linewidth = linewidth_raw;
    var v_line_color = line_color_in.line_color;
    if (v_linewidth < 1.0) {
        v_line_color.a *= v_linewidth;
        v_linewidth = 1.0;
    }

    // Calculate vertex position - annulus needs a square bounding box
    // Note: rect_geometry uses [-0.5, 0.5] range, so multiply by 2 to get [-1, 1] for full coverage
    let half_size = enclosing_size(screen_outer_radius, v_linewidth);
    let v_coords = vertex.position * half_size * 2.0;

    // Position without rotation
    var pos = screen_center + v_coords;

    // Convert to normalized device coordinates
    pos = pos + 0.5;
    pos = pos / uniforms.canvas_size;

    output.position = vec4f(2.0 * pos.x - 1.0, 1.0 - 2.0 * pos.y, 0.0, 1.0);
    output.v_coords = v_coords;
    output.v_inner_radius = screen_inner_radius;
    output.v_outer_radius = screen_outer_radius;
    output.v_linewidth = v_linewidth;
    output.v_line_color = v_line_color;
    output.v_fill_color = fill_color_in.fill_color;
    output.v_line_cap = line_cap;
    output.v_line_join = line_join;

    return output;
}

// ============================================================================
// SDF Functions for Annulus
// ============================================================================

// Circle SDF
fn circle_sdf(p: vec2f, radius: f32) -> f32 {
    return length(p) - radius;
}

// Annulus SDF (ring/donut shape)
fn annulus_sdf(p: vec2f, outer_radius: f32, inner_radius: f32) -> f32 {
    let outer = circle_sdf(p, outer_radius);
    let inner = circle_sdf(p, inner_radius);
    // Subtract inner from outer: max(outer, -inner)
    return max(outer, -inner);
}

// ============================================================================
// Fragment Shader Utilities
// ============================================================================

fn distance_to_alpha(dist: f32) -> f32 {
    return 1.0 - smoothstep(-0.5 * uniforms.antialias, 0.5 * uniforms.antialias, dist);
}

fn premultiply_alpha(color: vec4f, alpha: f32) -> vec4f {
    let a = color.a * alpha;
    return vec4f(color.rgb * a, a);
}

fn blend_over(src: vec4f, dst: vec4f) -> vec4f {
    return src + (1.0 - src.a) * dst;
}

fn fill_fraction(dist: f32) -> f32 {
    return distance_to_alpha(dist);
}

fn line_fraction(dist: f32, linewidth: f32) -> f32 {
    return distance_to_alpha(abs(dist) - 0.5 * linewidth);
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
    let dist = annulus_sdf(input.v_coords, input.v_outer_radius, input.v_inner_radius);

    // Calculate fill contribution
    let fill_frac = fill_fraction(dist);
    var color = premultiply_alpha(input.v_fill_color, fill_frac);

    // Calculate line/stroke contribution
    let line_frac = line_fraction(dist, input.v_linewidth);

    if (line_frac > 0.0) {
        let line_color = premultiply_alpha(input.v_line_color, line_frac);
        color = blend_over(line_color, color);
    }

    return color;
}
