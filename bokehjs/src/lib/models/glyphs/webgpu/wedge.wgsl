// Wedge shader for WebGPU rendering
// Implements a pie-chart slice shape (circle sector)

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

struct RadiusInput {
    @location(2) radius: f32,
}

struct AnglesInput {
    @location(3) angles: vec2f,  // start_angle, end_angle
}

struct LinePropsInput {
    @location(4) line_props: vec4f,  // linewidth, cap, join, show
}

struct LineColorInput {
    @location(5) line_color: vec4f,
}

struct FillColorInput {
    @location(6) fill_color: vec4f,
}

// Data passed from vertex to fragment shader
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) v_coords: vec2f,
    @location(1) v_radius: f32,
    @location(2) v_start_angle: f32,
    @location(3) v_end_angle: f32,
    @location(4) v_linewidth: f32,
    @location(5) v_line_color: vec4f,
    @location(6) v_fill_color: vec4f,
    @location(7) v_line_cap: f32,
    @location(8) v_line_join: f32,
}

// Calculate enclosing size for the wedge including line width and antialiasing
fn enclosing_size(radius: f32, linewidth: f32) -> f32 {
    return radius + linewidth + uniforms.antialias;
}

@vertex
fn vertex_main(
    vertex: VertexInput,
    pos_in: PositionInput,
    radius_in: RadiusInput,
    angles_in: AnglesInput,
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

    // Early exit for hidden markers
    if (show < 0.5 || radius_in.radius < 0.0) {
        output.position = vec4f(-2.0, -2.0, 0.0, 1.0);
        return output;
    }

    // Transform data coordinates to screen coordinates
    let screen_center = vec2f(
        pos_in.center.x * uniforms.x_scale + uniforms.x_offset,
        pos_in.center.y * uniforms.y_scale + uniforms.y_offset
    );

    // Transform radius from data units to screen units
    let screen_radius = radius_in.radius * uniforms.radius_scale;

    // Adjust line color alpha for thin lines
    var v_linewidth = linewidth_raw;
    var v_line_color = line_color_in.line_color;
    if (v_linewidth < 1.0) {
        v_line_color.a *= v_linewidth;
        v_linewidth = 1.0;
    }

    // Calculate vertex position - wedge needs a square bounding box
    // Note: rect_geometry uses [-0.5, 0.5] range, so multiply by 2 to get [-1, 1] for full coverage
    let half_size = enclosing_size(screen_radius, v_linewidth);
    let v_coords = vertex.position * half_size * 2.0;

    // Position without rotation (wedge angles handle orientation)
    var pos = screen_center + v_coords;

    // Convert to normalized device coordinates
    pos = pos + 0.5;
    pos = pos / uniforms.canvas_size;

    output.position = vec4f(2.0 * pos.x - 1.0, 1.0 - 2.0 * pos.y, 0.0, 1.0);
    output.v_coords = v_coords;
    output.v_radius = screen_radius;
    output.v_start_angle = angles_in.angles.x;
    output.v_end_angle = angles_in.angles.y;
    output.v_linewidth = v_linewidth;
    output.v_line_color = v_line_color;
    output.v_fill_color = fill_color_in.fill_color;
    output.v_line_cap = line_cap;
    output.v_line_join = line_join;

    return output;
}

// ============================================================================
// SDF Functions for Wedge
// ============================================================================

// Cross product z-component (for 2D)
fn cross_z(v0: vec2f, v1: vec2f) -> f32 {
    return v0.x * v1.y - v0.y * v1.x;
}

// Unit vector from angle
fn xy_from_angle(angle: f32) -> vec2f {
    return vec2f(cos(angle), sin(angle));
}

// Squared distance from point to segment
fn segment_square(p: vec2f, q: vec2f) -> f32 {
    let v = p - q * clamp(dot(p, q) / dot(q, q), 0.0, 1.0);
    return dot(v, v);
}

// Circle SDF
fn circle_sdf(p: vec2f, radius: f32) -> f32 {
    return length(p) - radius;
}

// Wedge SDF (from https://www.shadertoy.com/view/wldXWB - MIT licensed)
fn wedge_sdf(p: vec2f, r: f32, start_angle: f32, end_angle: f32) -> f32 {
    let a = r * xy_from_angle(start_angle);
    let b = r * xy_from_angle(end_angle);

    // Distance
    let d = sqrt(min(segment_square(p, a), segment_square(p, b)));

    // Sign
    var s: f32;
    if (cross_z(a, b) < 0.0) {
        s = sign(max(cross_z(a, p), cross_z(p, b)));
    } else {
        s = -sign(max(cross_z(p, a), cross_z(b, p)));
    }

    return s * d;
}

// Intersection of two SDFs
fn intersect_sdf(d1: f32, d2: f32) -> f32 {
    return max(d1, d2);
}

// Full wedge (pie slice) SDF = intersection of circle and wedge
fn full_wedge_sdf(p: vec2f, radius: f32, start_angle: f32, end_angle: f32) -> f32 {
    return intersect_sdf(
        circle_sdf(p, radius),
        wedge_sdf(p, radius, start_angle, end_angle)
    );
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
    let dist = full_wedge_sdf(input.v_coords, input.v_radius, input.v_start_angle, input.v_end_angle);

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
