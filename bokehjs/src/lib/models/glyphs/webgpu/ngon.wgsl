// N-gon (regular polygon) shader for WebGPU rendering
// Implements a regular polygon with n sides

// Constants
const PI: f32 = 3.14159265358979323846;
const SQRT2: f32 = 1.4142135623730951;

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

struct GeometryInput {
    @location(3) geometry: vec2f,  // angle, n (number of sides)
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
    @location(2) v_n: f32,
    @location(3) v_linewidth: f32,
    @location(4) v_line_color: vec4f,
    @location(5) v_fill_color: vec4f,
    @location(6) v_line_cap: f32,
    @location(7) v_line_join: f32,
}

// Calculate enclosing size including line width and antialiasing
fn enclosing_size(radius: f32, linewidth: f32) -> f32 {
    return radius + linewidth + uniforms.antialias;
}

@vertex
fn vertex_main(
    vertex: VertexInput,
    pos_in: PositionInput,
    radius_in: RadiusInput,
    geom_in: GeometryInput,
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

    let angle = geom_in.geometry.x;
    let n = geom_in.geometry.y;

    // Early exit for hidden markers or invalid ngons (need at least 3 sides)
    if (show < 0.5 || radius_in.radius < 0.0 || n < 3.0) {
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

    // Calculate vertex position
    // Note: rect_geometry uses [-0.5, 0.5] range, so multiply by 2 to get [-1, 1] for full coverage
    let half_size = enclosing_size(screen_radius, v_linewidth);
    var v_coords = vertex.position * half_size * 2.0;

    // Apply rotation
    let c = cos(-angle);
    let s = sin(-angle);
    let rotated_coords = mat2x2f(c, -s, s, c) * v_coords;
    var pos = screen_center + rotated_coords;

    // Convert to normalized device coordinates
    pos = pos + 0.5;
    pos = pos / uniforms.canvas_size;

    output.position = vec4f(2.0 * pos.x - 1.0, 1.0 - 2.0 * pos.y, 0.0, 1.0);
    output.v_coords = v_coords;
    output.v_radius = screen_radius;
    output.v_n = n;
    output.v_linewidth = v_linewidth;
    output.v_line_color = v_line_color;
    output.v_fill_color = fill_color_in.fill_color;
    output.v_line_cap = line_cap;
    output.v_line_join = line_join;

    return output;
}

// ============================================================================
// SDF Functions for N-gon
// ============================================================================

fn line_join_distance_no_miter(
    p: vec2f, corner: vec2f, unit_normal: vec2f, offset: f32, line_join: i32, linewidth: f32
) -> f32 {
    let dist_outside = dot(p - corner, unit_normal) - offset;
    if (line_join == JOIN_BEVEL && dist_outside > -0.5 * uniforms.antialias) {
        return dist_outside + 0.5 * linewidth;
    } else if (dist_outside > 0.0) {
        return distance(p, corner);
    }
    return -linewidth - uniforms.antialias;
}

// N-gon SDF
fn ngon_sdf(p_in: vec2f, radius: f32, n: f32, line_join: i32, linewidth: f32) -> f32 {
    let side_angle = 2.0 * PI / n;  // Angle subtended by 1 side of ngon at center

    // Use symmetry to transform p around center into first half of first side of ngon
    var p = vec2f(p_in.x, -p_in.y);
    var angle = atan2(p.x, p.y);
    // Manual modulo that handles negative values correctly
    angle = angle - side_angle * floor(angle / side_angle);
    angle = min(angle, side_angle - angle);
    p = length(p) * vec2f(sin(angle), cos(angle));

    let half_angle = 0.5 * side_angle;
    let cos_half_angle = cos(half_angle);
    let unit_normal = vec2f(sin(half_angle), cos_half_angle);
    let corner = vec2f(0.0, radius);
    var dist = dot(p - corner, unit_normal);

    if (line_join != JOIN_MITER) {
        dist = max(dist, line_join_distance_no_miter(
            p, corner, vec2f(0.0, 1.0), 0.5 * linewidth * cos_half_angle, line_join, linewidth));
    }

    return dist;
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
    let line_join = i32(input.v_line_join + 0.5);
    let dist = ngon_sdf(input.v_coords, input.v_radius, input.v_n, line_join, input.v_linewidth);

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
