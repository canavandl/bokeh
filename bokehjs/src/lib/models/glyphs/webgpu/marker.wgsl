// Marker shader for WebGPU rendering
// Supports all MarkerType variants

// Constants
const SQRT2: f32 = 1.4142135623730951;
const SQRT3: f32 = 1.7320508075688772;
const SQRT5: f32 = 2.23606797749979;
const SQRT13: f32 = 3.605551275463989;
const PI: f32 = 3.14159265358979323846;

// Marker type constants (set via pipeline specialization)
// These must match the order in MarkerType enum
override MARKER_ASTERISK: bool = false;
override MARKER_CIRCLE: bool = false;
override MARKER_CIRCLE_CROSS: bool = false;
override MARKER_CIRCLE_DOT: bool = false;
override MARKER_CIRCLE_X: bool = false;
override MARKER_CIRCLE_Y: bool = false;
override MARKER_CROSS: bool = false;
override MARKER_DASH: bool = false;
override MARKER_DIAMOND: bool = false;
override MARKER_DIAMOND_CROSS: bool = false;
override MARKER_DIAMOND_DOT: bool = false;
override MARKER_DOT: bool = false;
override MARKER_HEX: bool = false;
override MARKER_HEX_DOT: bool = false;
override MARKER_INVERTED_TRIANGLE: bool = false;
override MARKER_PLUS: bool = false;
override MARKER_SQUARE: bool = false;
override MARKER_SQUARE_CROSS: bool = false;
override MARKER_SQUARE_DOT: bool = false;
override MARKER_SQUARE_PIN: bool = false;
override MARKER_SQUARE_X: bool = false;
override MARKER_STAR: bool = false;
override MARKER_STAR_DOT: bool = false;
override MARKER_TRIANGLE: bool = false;
override MARKER_TRIANGLE_DOT: bool = false;
override MARKER_TRIANGLE_PIN: bool = false;
override MARKER_X: bool = false;
override MARKER_Y: bool = false;
// Additional types used by specific glyphs
override MARKER_RECT: bool = false;
override MARKER_ROUND_RECT: bool = false;

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
    size_hint: f32,
    border_radius: vec4f,
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

// Instance data split into separate buffers for efficient partial updates
struct PositionInput {
    @location(1) center: vec2f,
}

struct SizeInput {
    @location(2) size: vec2f,
}

struct GeometryInput {
    @location(3) angle_aux: vec2f,
}

struct LinePropsInput {
    @location(4) line_props: vec4f,
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
    @location(1) v_size: vec2f,
    @location(2) v_linewidth: f32,
    @location(3) v_line_color: vec4f,
    @location(4) v_fill_color: vec4f,
    @location(5) v_line_cap: f32,
    @location(6) v_line_join: f32,
}

// Calculate enclosing size for a marker including line width and antialiasing
fn enclosing_size(size: vec2f, linewidth: f32) -> vec2f {
    return size + linewidth + uniforms.antialias;
}

// Check if this is a line-only marker (no fill)
fn is_line_only() -> bool {
    return MARKER_ASTERISK || MARKER_CROSS || MARKER_DASH || MARKER_DOT || MARKER_X || MARKER_Y;
}

@vertex
fn vertex_main(
    vertex: VertexInput,
    pos_in: PositionInput,
    size_in: SizeInput,
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

    // Early exit for hidden markers
    if (show < 0.5 || size_in.size.x < 0.0 || size_in.size.y < 0.0) {
        output.position = vec4f(-2.0, -2.0, 0.0, 1.0);
        return output;
    }

    // Transform data coordinates to screen coordinates
    let screen_center = vec2f(
        pos_in.center.x * uniforms.x_scale + uniforms.x_offset,
        pos_in.center.y * uniforms.y_scale + uniforms.y_offset
    );

    // Transform size from data units to screen units
    let screen_size = size_in.size * uniforms.radius_scale;

    // Determine marker size based on type
    var v_size: vec2f;
    if (MARKER_CIRCLE || MARKER_CIRCLE_CROSS || MARKER_CIRCLE_DOT || MARKER_CIRCLE_X ||
        MARKER_CIRCLE_Y || MARKER_ASTERISK || MARKER_CROSS || MARKER_DASH || MARKER_DOT ||
        MARKER_X || MARKER_Y || MARKER_PLUS || MARKER_DIAMOND || MARKER_DIAMOND_CROSS ||
        MARKER_DIAMOND_DOT || MARKER_HEX || MARKER_HEX_DOT || MARKER_SQUARE ||
        MARKER_SQUARE_CROSS || MARKER_SQUARE_DOT || MARKER_SQUARE_PIN || MARKER_SQUARE_X ||
        MARKER_STAR || MARKER_STAR_DOT || MARKER_TRIANGLE || MARKER_TRIANGLE_DOT ||
        MARKER_TRIANGLE_PIN || MARKER_INVERTED_TRIANGLE) {
        // Square markers use width for both dimensions
        v_size = vec2f(screen_size.x, screen_size.x);
    } else {
        // Rect uses width and height
        v_size = screen_size;
    }

    // Adjust line color alpha for thin lines
    var v_linewidth = linewidth_raw;
    var v_line_color = line_color_in.line_color;
    if (v_linewidth < 1.0) {
        v_line_color.a *= v_linewidth;
        v_linewidth = 1.0;
    }

    // Calculate vertex position in local marker space
    let v_coords = vertex.position * enclosing_size(v_size, v_linewidth);

    // Apply rotation
    let angle = geom_in.angle_aux.x;
    let c = cos(-angle);
    let s = sin(-angle);
    let rotation = mat2x2f(c, -s, s, c);
    var pos = screen_center + rotation * v_coords;

    // Convert to normalized device coordinates
    pos = pos + 0.5;
    pos = pos / uniforms.canvas_size;

    output.position = vec4f(2.0 * pos.x - 1.0, 1.0 - 2.0 * pos.y, 0.0, 1.0);
    output.v_coords = v_coords;
    output.v_size = v_size;
    output.v_linewidth = v_linewidth;
    output.v_line_color = v_line_color;
    output.v_fill_color = fill_color_in.fill_color;
    output.v_line_cap = line_cap;
    output.v_line_join = line_join;

    return output;
}

// ============================================================================
// SDF Helper Functions
// ============================================================================

fn end_cap_distance(p: vec2f, end_point: vec2f, unit_direction: vec2f, line_cap: i32, linewidth: f32) -> f32 {
    let offset = p - end_point;
    if (line_cap == CAP_BUTT) {
        return dot(offset, unit_direction) + 0.5 * linewidth;
    } else if (line_cap == CAP_SQUARE) {
        return dot(offset, unit_direction);
    } else if (line_cap == CAP_ROUND && dot(offset, unit_direction) > 0.0) {
        return length(offset);
    }
    return -linewidth - uniforms.antialias;
}

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

fn one_cross(p_in: vec2f, line_cap: i32, len: f32, linewidth: f32) -> f32 {
    var p = abs(p_in);
    if (p.y > p.x) {
        p = p.yx;
    }
    let dist = p.y;
    let end_dist = end_cap_distance(p, vec2f(len, 0.0), vec2f(1.0, 0.0), line_cap, linewidth);
    return max(dist, end_dist);
}

fn one_y(p_in: vec2f, line_cap: i32, len: f32, linewidth: f32) -> f32 {
    var p = vec2f(abs(p_in.x), -p_in.y);

    let k = 6.0 / sqrt(13.0);
    let unit_along = vec2f(0.5 * k, k / 3.0);
    let end_point = vec2f(0.5 * len * SQRT3, len * SQRT3 / 3.0);
    var dist = max(abs(dot(p, vec2f(-unit_along.y, unit_along.x))),
                   end_cap_distance(p, end_point, unit_along, line_cap, linewidth));

    if (p.y < 0.0) {
        let vert_dist = max(p.x,
                            end_cap_distance(p, vec2f(0.0, -len), vec2f(0.0, -1.0), line_cap, linewidth));
        dist = min(dist, vert_dist);
    }
    return dist;
}

// ============================================================================
// Marker SDF Functions
// ============================================================================

fn asterisk_sdf(p: vec2f, size: vec2f, line_cap: i32, linewidth: f32) -> f32 {
    let p_diag = vec2f((p.x + p.y) / SQRT2, (p.x - p.y) / SQRT2);
    let len = 0.5 * size.x;
    return min(one_cross(p, line_cap, len, linewidth),
               one_cross(p_diag, line_cap, len, linewidth));
}

fn circle_sdf(p: vec2f, size: vec2f) -> f32 {
    return length(p) - 0.5 * size.x;
}

fn cross_sdf(p: vec2f, size: vec2f, line_cap: i32, linewidth: f32) -> f32 {
    return one_cross(p, line_cap, 0.5 * size.x, linewidth);
}

fn dash_sdf(p: vec2f, size: vec2f, line_cap: i32, linewidth: f32) -> f32 {
    let pa = abs(p);
    let dist = pa.y;
    let end_dist = end_cap_distance(pa, vec2f(0.5 * size.x, 0.0), vec2f(1.0, 0.0), line_cap, linewidth);
    return max(dist, end_dist);
}

fn diamond_sdf(p: vec2f, size: vec2f, line_join: i32, linewidth: f32) -> f32 {
    let pa = abs(p);
    let r = 0.5 * size.x;
    var dist = dot(pa, vec2f(3.0, 2.0)) / SQRT13 - 2.0 * r / SQRT13;

    if (line_join != JOIN_MITER) {
        dist = max(dist, line_join_distance_no_miter(
            pa, vec2f(0.0, r), vec2f(0.0, 1.0), linewidth / SQRT13, line_join, linewidth));
        dist = max(dist, line_join_distance_no_miter(
            pa, vec2f(r * 2.0 / 3.0, 0.0), vec2f(1.0, 0.0), linewidth * (1.5 / SQRT13), line_join, linewidth));
    }
    return dist;
}

fn dot_sdf() -> f32 {
    // Dot is always appended
    return 1000.0;  // Large value means no main shape
}

fn hex_sdf(p: vec2f, size: vec2f, line_join: i32, linewidth: f32) -> f32 {
    let pa = abs(p);
    let rx = size.x / 2.0;
    let h = size.y * (SQRT3 / 4.0);
    let len_normal = sqrt(h * h + 0.25 * rx * rx);
    let unit_normal = vec2f(h, 0.5 * rx) / len_normal;
    var dist = max(dot(pa, unit_normal) - rx * h / len_normal, pa.y - h);

    if (line_join != JOIN_MITER) {
        dist = max(dist, line_join_distance_no_miter(
            pa, vec2f(rx, 0.0), vec2f(1.0, 0.0), 0.5 * linewidth * unit_normal.x, line_join, linewidth));
        let unit_normal2 = normalize(unit_normal + vec2f(0.0, 1.0));
        dist = max(dist, line_join_distance_no_miter(
            pa, vec2f(0.5 * rx, h), unit_normal2, 0.5 * linewidth * unit_normal2.y, line_join, linewidth));
    }
    return dist;
}

fn plus_sdf(p: vec2f, size: vec2f, line_join: i32, linewidth: f32) -> f32 {
    var pa = abs(p);
    if (pa.y > pa.x) {
        pa = pa.yx;
    }

    let r = 0.5 * size.x;
    let offset = pa - vec2f(r, 0.375 * r);
    var dist = max(offset.x, offset.y);

    if (line_join != JOIN_MITER) {
        dist = max(dist, line_join_distance_no_miter(
            offset, vec2f(0.0, 0.0), vec2f(1.0 / SQRT2, 1.0 / SQRT2), linewidth / (2.0 * SQRT2), line_join, linewidth));
        dist = min(dist, -line_join_distance_no_miter(
            offset, vec2f(-5.0 * r / 8.0, 0.0), vec2f(-1.0 / SQRT2, -1.0 / SQRT2), linewidth / (2.0 * SQRT2), line_join, linewidth));
    }
    return dist;
}

fn rect_sdf(p: vec2f, size: vec2f) -> f32 {
    let d = abs(p) - size / 2.0;
    return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}

fn square_sdf(p: vec2f, size: vec2f, line_join: i32, linewidth: f32) -> f32 {
    let p2 = abs(p) - size / 2.0;
    var dist = max(p2.x, p2.y);

    if (line_join != JOIN_MITER) {
        dist = max(dist, line_join_distance_no_miter(
            p2, vec2f(0.0, 0.0), vec2f(1.0 / SQRT2, 1.0 / SQRT2), linewidth / (2.0 * SQRT2), line_join, linewidth));
    }
    return dist;
}

fn square_pin_sdf(p: vec2f, size: vec2f, line_join: i32, linewidth: f32) -> f32 {
    var pa = abs(p);
    if (pa.y > pa.x) {
        pa = pa.yx;
    }

    let r = 0.5 * size.x;
    let center_x = r * 2.44375;
    let radius = r * 1.75626;
    var dist = radius - distance(pa, vec2f(center_x, 0.0));

    // Simplified join handling
    if (line_join != JOIN_MITER) {
        let corner_dist = line_join_distance_no_miter(
            pa, vec2f(r, r), vec2f(1.0 / SQRT2, 1.0 / SQRT2), linewidth * 0.1124, line_join, linewidth);
        dist = max(dist, corner_dist);
    }
    return dist;
}

fn star_sdf(p: vec2f, size: vec2f, line_join: i32, linewidth: f32) -> f32 {
    let COS72 = 0.25 * (SQRT5 - 1.0);
    let SIN72 = sqrt((5.0 + SQRT5) / 8.0);

    var angle = atan2(p.x, p.y);
    angle = (angle % (0.4 * PI)) - 0.2 * PI;
    if (angle < -0.2 * PI) { angle += 0.4 * PI; }
    let pa = length(p) * vec2f(cos(angle), abs(sin(angle)));

    let r = 0.5 * size.x;
    let a = r * sqrt(5.0 - 2.0 * SQRT5);
    var dist = dot(pa, vec2f(COS72, SIN72)) - r * COS72;

    if (line_join != JOIN_MITER) {
        dist = max(dist, line_join_distance_no_miter(
            pa, vec2f(r, 0.0), vec2f(1.0, 0.0), linewidth * (0.5 * COS72), line_join, linewidth));
        let COS36 = sqrt(0.5 + COS72 / 2.0);
        let SIN36 = sqrt(0.5 - COS72 / 2.0);
        dist = min(dist, -line_join_distance_no_miter(
            pa, vec2f(r - a * SIN72, a * COS72), vec2f(-COS36, -SIN36), linewidth * (0.5 * COS36), line_join, linewidth));
    }
    return dist;
}

fn triangle_sdf(p: vec2f, size: vec2f, line_join: i32, linewidth: f32, inverted: bool) -> f32 {
    let r = 0.5 * size.x;
    let a = r * SQRT3 / 3.0;

    var pa: vec2f;
    if (inverted) {
        pa = vec2f(abs(p.x), -p.y);
    } else {
        pa = vec2f(abs(p.x), p.y);
    }

    var dist = max(0.5 * dot(pa, vec2f(SQRT3, -1.0)) - a, pa.y - a);

    if (line_join != JOIN_MITER) {
        dist = max(dist, line_join_distance_no_miter(
            pa, vec2f(0.0, -(2.0 / SQRT3) * r), vec2f(0.0, -1.0), linewidth * 0.25, line_join, linewidth));
        dist = max(dist, line_join_distance_no_miter(
            pa, vec2f(r, a), vec2f(SQRT3 / 2.0, 0.5), linewidth * 0.25, line_join, linewidth));
    }
    return dist;
}

fn triangle_pin_sdf(p: vec2f, size: vec2f, line_join: i32, linewidth: f32) -> f32 {
    var angle = atan2(p.x, -p.y);
    angle = (angle % (PI * 2.0 / 3.0)) - PI / 3.0;
    if (angle < -PI / 3.0) { angle += PI * 2.0 / 3.0; }
    let pa = length(p) * vec2f(cos(angle), abs(sin(angle)));

    let a_const = 1.0 / SQRT3;
    let b_const = SQRT3 / 8.0;
    let c_const = (a_const - b_const) / 2.0;
    let x_const = (1.0 - c_const * c_const) / (2.0 * c_const);
    let center_x = x_const + a_const;
    let radius = x_const + c_const;
    let r = 0.5 * size.x;
    var dist = r * radius - distance(pa, vec2f(r * center_x, 0.0));

    if (line_join != JOIN_MITER) {
        let corner_dist = line_join_distance_no_miter(
            pa, vec2f(a_const * r, r), vec2f(0.5, 0.5 * SQRT3), linewidth * 0.0882, line_join, linewidth);
        dist = max(dist, corner_dist);
    }
    return dist;
}

fn x_sdf(p: vec2f, size: vec2f, line_cap: i32, linewidth: f32) -> f32 {
    let p_rot = vec2f((p.x + p.y) / SQRT2, (p.x - p.y) / SQRT2);
    return one_cross(p_rot, line_cap, 0.5 * size.x, linewidth);
}

fn y_sdf(p: vec2f, size: vec2f, line_cap: i32, linewidth: f32) -> f32 {
    return one_y(p, line_cap, 0.5 * size.x, linewidth);
}

fn round_rect_sdf(p: vec2f, size: vec2f, radius: vec4f) -> f32 {
    var r: vec2f;
    if (p.x > 0.0) {
        if (p.y > 0.0) {
            r = vec2f(radius.x, radius.x);
        } else {
            r = vec2f(radius.y, radius.y);
        }
    } else {
        if (p.y > 0.0) {
            r = vec2f(radius.w, radius.w);
        } else {
            r = vec2f(radius.z, radius.z);
        }
    }
    let q = abs(p) - size / 2.0 + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2f(0.0))) - r.x;
}

// Main marker SDF dispatch
fn marker_sdf(p: vec2f, size: vec2f, line_cap: i32, line_join: i32, linewidth: f32) -> f32 {
    if (MARKER_ASTERISK) {
        return asterisk_sdf(p, size, line_cap, linewidth);
    } else if (MARKER_CIRCLE || MARKER_CIRCLE_CROSS || MARKER_CIRCLE_DOT || MARKER_CIRCLE_X || MARKER_CIRCLE_Y) {
        return circle_sdf(p, size);
    } else if (MARKER_CROSS) {
        return cross_sdf(p, size, line_cap, linewidth);
    } else if (MARKER_DASH) {
        return dash_sdf(p, size, line_cap, linewidth);
    } else if (MARKER_DIAMOND || MARKER_DIAMOND_CROSS || MARKER_DIAMOND_DOT) {
        return diamond_sdf(p, size, line_join, linewidth);
    } else if (MARKER_DOT) {
        return dot_sdf();
    } else if (MARKER_HEX || MARKER_HEX_DOT) {
        return hex_sdf(p, size, line_join, linewidth);
    } else if (MARKER_INVERTED_TRIANGLE) {
        return triangle_sdf(p, size, line_join, linewidth, true);
    } else if (MARKER_PLUS) {
        return plus_sdf(p, size, line_join, linewidth);
    } else if (MARKER_RECT) {
        return rect_sdf(p, size);
    } else if (MARKER_ROUND_RECT) {
        return round_rect_sdf(p, size, uniforms.border_radius);
    } else if (MARKER_SQUARE || MARKER_SQUARE_CROSS || MARKER_SQUARE_DOT || MARKER_SQUARE_X) {
        return square_sdf(p, size, line_join, linewidth);
    } else if (MARKER_SQUARE_PIN) {
        return square_pin_sdf(p, size, line_join, linewidth);
    } else if (MARKER_STAR || MARKER_STAR_DOT) {
        return star_sdf(p, size, line_join, linewidth);
    } else if (MARKER_TRIANGLE || MARKER_TRIANGLE_DOT) {
        return triangle_sdf(p, size, line_join, linewidth, false);
    } else if (MARKER_TRIANGLE_PIN) {
        return triangle_pin_sdf(p, size, line_join, linewidth);
    } else if (MARKER_X) {
        return x_sdf(p, size, line_cap, linewidth);
    } else if (MARKER_Y) {
        return y_sdf(p, size, line_cap, linewidth);
    }
    // Default to circle
    return circle_sdf(p, size);
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

fn dot_fraction(p: vec2f, size: vec2f) -> f32 {
    let radius = 0.125 * size.x;
    let dot_distance = max(length(p) - radius, -0.5 * uniforms.antialias);
    return fill_fraction(dot_distance);
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
    let line_cap = i32(input.v_line_cap + 0.5);
    let line_join = i32(input.v_line_join + 0.5);

    let dist = marker_sdf(input.v_coords, input.v_size, line_cap, line_join, input.v_linewidth);

    // Calculate fill contribution
    var color: vec4f;
    if (is_line_only()) {
        color = vec4f(0.0, 0.0, 0.0, 0.0);
    } else {
        let fill_frac = fill_fraction(dist);
        color = premultiply_alpha(input.v_fill_color, fill_frac);
    }

    // Calculate line/stroke contribution
    var line_frac = line_fraction(dist, input.v_linewidth);

    // Handle appended decorations (dot, cross, x, y)
    if (MARKER_DOT || MARKER_CIRCLE_DOT || MARKER_DIAMOND_DOT || MARKER_HEX_DOT ||
        MARKER_SQUARE_DOT || MARKER_STAR_DOT || MARKER_TRIANGLE_DOT) {
        line_frac = max(line_frac, dot_fraction(input.v_coords, input.v_size));
    }

    if (MARKER_CIRCLE_CROSS || MARKER_SQUARE_CROSS) {
        let cross_dist = one_cross(input.v_coords, line_cap, 0.5 * input.v_size.x, input.v_linewidth);
        line_frac = max(line_frac, line_fraction(cross_dist, input.v_linewidth));
    }

    if (MARKER_DIAMOND_CROSS) {
        // Diamond cross has different arm lengths
        var pa = abs(input.v_coords);
        let switch_xy = pa.y > pa.x;
        if (switch_xy) { pa = pa.yx; }
        let len = select(input.v_size.x / 3.0, input.v_size.x / 2.0, switch_xy);
        let cross_dist = max(pa.y, end_cap_distance(pa, vec2f(len, 0.0), vec2f(1.0, 0.0), line_cap, input.v_linewidth));
        line_frac = max(line_frac, line_fraction(cross_dist, input.v_linewidth));
    }

    if (MARKER_CIRCLE_X || MARKER_SQUARE_X) {
        let x_len = select(input.v_size.x / SQRT2, 0.5 * input.v_size.x, MARKER_CIRCLE_X);
        let p_rot = vec2f((input.v_coords.x + input.v_coords.y) / SQRT2,
                          (input.v_coords.x - input.v_coords.y) / SQRT2);
        let x_dist = one_cross(p_rot, line_cap, x_len, input.v_linewidth);
        line_frac = max(line_frac, line_fraction(x_dist, input.v_linewidth));
    }

    if (MARKER_CIRCLE_Y) {
        let y_dist = one_y(input.v_coords, line_cap, 0.5 * input.v_size.x, input.v_linewidth);
        line_frac = max(line_frac, line_fraction(y_dist, input.v_linewidth));
    }

    if (line_frac > 0.0) {
        let line_color = premultiply_alpha(input.v_line_color, line_frac);
        color = blend_over(line_color, color);
    }

    return color;
}
