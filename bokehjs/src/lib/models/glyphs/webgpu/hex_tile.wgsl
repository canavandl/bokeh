// Hex tile shader for WebGPU rendering
// Implements hexagon tiles with SDF-based rendering

// Constants
const PI: f32 = 3.14159265358979323846;
const SQRT3: f32 = 1.7320508075688772;

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
    // Hex tile dimensions (derived from svx/svy)
    hex_width: f32,   // width of hexagon
    hex_height: f32,  // height of hexagon
    // Rotation angle (0 for flattop, PI/2 for pointytop)
    rotation: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Vertex input from quad geometry (per-vertex)
struct VertexInput {
    @location(0) position: vec2f,
}

// Instance data buffers
struct PositionInput {
    @location(1) center: vec2f,  // Screen coordinates (sx, sy)
}

struct ScaleInput {
    @location(2) scale: f32,  // Per-tile scale factor
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
    @location(1) v_size: vec2f,
    @location(2) v_linewidth: f32,
    @location(3) v_line_color: vec4f,
    @location(4) v_fill_color: vec4f,
    @location(5) v_line_cap: f32,
    @location(6) v_line_join: f32,
}

// Calculate enclosing size including line width and antialiasing
fn enclosing_size(hex_size: vec2f, linewidth: f32) -> vec2f {
    let half_size = hex_size * 0.5;
    return half_size + linewidth + uniforms.antialias;
}

@vertex
fn vertex_main(
    vertex: VertexInput,
    pos_in: PositionInput,
    scale_in: ScaleInput,
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

    let scale = scale_in.scale;

    // Early exit for hidden markers
    if (show < 0.5 || scale <= 0.0) {
        output.position = vec4f(-2.0, -2.0, 0.0, 1.0);
        return output;
    }

    // Screen position is already provided directly
    let screen_center = pos_in.center;

    // Scaled hex size
    let hex_size = vec2f(uniforms.hex_width, uniforms.hex_height) * scale;

    // Adjust line color alpha for thin lines
    var v_linewidth = linewidth_raw;
    var v_line_color = line_color_in.line_color;
    if (v_linewidth < 1.0) {
        v_line_color.a *= v_linewidth;
        v_linewidth = 1.0;
    }

    // Calculate vertex position
    let half_size = enclosing_size(hex_size, v_linewidth);

    // Apply rotation for orientation
    let c = cos(uniforms.rotation);
    let s = sin(uniforms.rotation);
    let rot_matrix = mat2x2f(c, -s, s, c);

    // Rotate the quad vertex positions
    var v_coords = vertex.position * half_size * 2.0;
    let rotated_coords = rot_matrix * v_coords;
    var pos = screen_center + rotated_coords;

    // Convert to normalized device coordinates
    pos = pos + 0.5;
    pos = pos / uniforms.canvas_size;

    output.position = vec4f(2.0 * pos.x - 1.0, 1.0 - 2.0 * pos.y, 0.0, 1.0);
    output.v_coords = v_coords;
    output.v_size = hex_size;
    output.v_linewidth = v_linewidth;
    output.v_line_color = v_line_color;
    output.v_fill_color = fill_color_in.fill_color;
    output.v_line_cap = line_cap;
    output.v_line_join = line_join;

    return output;
}

// ============================================================================
// SDF Functions for Hexagon
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

// Hexagon SDF
// A regular hexagon has v_size.x == v_size.y = r where r is the length of
// each of the 3 sides of the 6 equilateral triangles that comprise the hex.
// Only consider +ve quadrant, the 3 corners are at (0, h), (rx/2, h), (rx, 0)
// where rx = 0.5*v_size.x and h = v_size.y*SQRT3/4.
fn hexagon_sdf(p_in: vec2f, size: vec2f, line_join: i32, linewidth: f32) -> f32 {
    // Use symmetry - work in positive quadrant
    var p = abs(p_in);

    let rx = size.x / 2.0;
    let h = size.y * (SQRT3 / 4.0);
    let len_normal = sqrt(h * h + 0.25 * rx * rx);
    let unit_normal = vec2f(h, 0.5 * rx) / len_normal;

    // Distance from sloping line and horizontal line
    var dist = max(
        dot(p, unit_normal) - rx * h / len_normal,
        p.y - h
    );

    if (line_join != JOIN_MITER) {
        // Corner at (rx, 0)
        dist = max(dist, line_join_distance_no_miter(
            p, vec2f(rx, 0.0), vec2f(1.0, 0.0), 0.5 * linewidth * unit_normal.x, line_join, linewidth));

        // Corner at (rx/2, h)
        let corner_normal = normalize(unit_normal + vec2f(0.0, 1.0));
        dist = max(dist, line_join_distance_no_miter(
            p, vec2f(0.5 * rx, h), corner_normal, 0.5 * linewidth * corner_normal.y, line_join, linewidth));
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
    let dist = hexagon_sdf(input.v_coords, input.v_size, line_join, input.v_linewidth);

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
