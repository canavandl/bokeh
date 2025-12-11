// WebGPU line shader for rendering line segments with joins and caps
// Adapted from WebGL regl_line shaders

// Line cap types
const CAP_BUTT: i32 = 0;
const CAP_ROUND: i32 = 1;
const CAP_SQUARE: i32 = 2;

// Line join types
const JOIN_MITER: i32 = 0;
const JOIN_ROUND: i32 = 1;
const JOIN_BEVEL: i32 = 2;

const SMALL: f32 = 1e-6;
const ONE_MINUS_SMALL: f32 = 1.0 - 1e-6;

struct Uniforms {
    canvas_width: f32,
    canvas_height: f32,
    antialias: f32,
    miter_limit: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec2f,           // Quad vertex position (-2 to 2, -1 to 1)
    @location(1) point_prev: vec2f,         // Previous point
    @location(2) point_start: vec2f,        // Segment start point
    @location(3) point_end: vec2f,          // Segment end point
    @location(4) point_next: vec2f,         // Next point
    @location(5) show_flags: vec4f,         // show_prev, show_curr, show_next, unused
    @location(6) line_props: vec4f,         // linewidth, cap, join, unused
    @location(7) line_color: vec4f,         // RGBA color
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) coords: vec2f,              // Local segment coordinates
    @location(1) line_color: vec4f,
    @location(2) line_params: vec4f,         // linewidth, cap, join, segment_length
    @location(3) turn_angles: vec2f,         // cos_turn_angle_start, cos_turn_angle_end
    @location(4) flags: f32,                 // Packed boolean flags
}

fn cross_z(v0: vec2f, v1: vec2f) -> f32 {
    return v0.x * v1.y - v0.y * v1.x;
}

fn right_vector(v: vec2f) -> vec2f {
    return vec2f(v.y, -v.x);
}

fn normalize_check_len(v: vec2f, len: f32) -> vec2f {
    if (abs(len) < SMALL) {
        return vec2f(1.0, 0.0);
    }
    return v / len;
}

fn normalize_check(v: vec2f) -> vec2f {
    return normalize_check_len(v, length(v));
}

// Calculate turn angle with adjacent segment
fn calc_turn_angle(has_cap: bool, segment_right: vec2f, other_right: vec2f) -> vec4f {
    // Returns: (cos_turn_angle, sin_turn_angle, point_right.x, point_right.y)
    var cos_turn_angle: f32;
    var sin_turn_angle: f32;
    var point_right: vec2f;

    let diff = segment_right + other_right;
    let len = length(diff);

    if (has_cap || len < SMALL) {
        point_right = segment_right;
        cos_turn_angle = -1.0;
        sin_turn_angle = 0.0;
    } else {
        point_right = diff / len;
        cos_turn_angle = dot(segment_right, other_right);
        sin_turn_angle = cross_z(segment_right, other_right);
    }

    return vec4f(cos_turn_angle, sin_turn_angle, point_right.x, point_right.y);
}

fn miter_too_large(join_type: i32, cos_turn_angle: f32, miter_limit: f32) -> bool {
    let cos_half_angle_sqr = 0.5 * (1.0 + cos_turn_angle);
    return join_type == JOIN_MITER && cos_half_angle_sqr < 1.0 / (miter_limit * miter_limit);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let show_prev = input.show_flags.x;
    let show_curr = input.show_flags.y;
    let show_next = input.show_flags.z;

    // Early exit for invalid segments
    if (show_curr < 0.5) {
        output.position = vec4f(-2.0, -2.0, 0.0, 1.0);
        return output;
    }

    var linewidth = input.line_props.x;
    let cap_type = i32(input.line_props.y + 0.5);
    let join_type = i32(input.line_props.z + 0.5);

    var color = input.line_color;
    if (linewidth < 1.0) {
        color.a *= linewidth;
        linewidth = 1.0;
    }

    let halfwidth = 0.5 * (linewidth + uniforms.antialias);

    let segment_vec = input.point_end - input.point_start;
    let segment_length = length(segment_vec);
    let segment_along = normalize_check_len(segment_vec, segment_length);
    let segment_right = right_vector(segment_along);

    let prev_along = normalize_check(input.point_start - input.point_prev);
    let prev_right = right_vector(prev_along);
    let next_right = right_vector(normalize_check(input.point_next - input.point_end));

    var coords = vec2f(0.0, input.position.y * halfwidth);

    let has_start_cap = show_prev < 0.5;
    let has_end_cap = show_next < 0.5;

    let turn_start = calc_turn_angle(has_start_cap, segment_right, prev_right);
    let turn_end = calc_turn_angle(has_end_cap, segment_right, next_right);

    let cos_turn_angle_start = turn_start.x;
    let sin_turn_angle_start = turn_start.y;
    let cos_turn_angle_end = turn_end.x;
    let sin_turn_angle_end = turn_end.y;

    let sign_turn_right_start = select(-1.0, 1.0, sin_turn_angle_start >= 0.0);

    let miter_too_large_start = !has_start_cap && miter_too_large(join_type, cos_turn_angle_start, uniforms.miter_limit);
    let miter_too_large_end = !has_end_cap && miter_too_large(join_type, cos_turn_angle_end, uniforms.miter_limit);

    let sign_at_start = -sign(input.position.x);
    let point = select(input.point_end, input.point_start, sign_at_start > 0.0);

    var xy: vec2f;

    if ((has_start_cap && sign_at_start > 0.0) || (has_end_cap && sign_at_start < 0.0)) {
        // Cap
        xy = point - segment_right * (halfwidth * input.position.y);
        if (cap_type == CAP_BUTT) {
            xy -= sign_at_start * 0.5 * uniforms.antialias * segment_along;
        } else {
            xy -= sign_at_start * halfwidth * segment_along;
        }
    } else if (sign_at_start > 0.0) {
        // Join at start
        let inside_point = input.point_start + segment_right * (sign_turn_right_start * halfwidth);
        let prev_outside_point = input.point_start - prev_right * (sign_turn_right_start * halfwidth);

        if (join_type == JOIN_ROUND || join_type == JOIN_BEVEL || miter_too_large_start) {
            if (cos_turn_angle_start <= 0.0) {
                xy = input.point_start - segment_right * (halfwidth * input.position.y) - halfwidth * segment_along;
            } else {
                if (input.position.x < -1.5) {
                    xy = prev_outside_point;
                    coords.y = -dot(xy - input.point_start, segment_right);
                } else if (input.position.y * sign_turn_right_start > 0.0) {
                    // Outside corner of turn
                    let d = halfwidth * abs(sin_turn_angle_start);
                    xy = input.point_start - segment_right * (halfwidth * input.position.y) - d * segment_along;
                } else {
                    xy = inside_point;
                }
            }
        } else {
            // Miter join
            if (input.position.x < -1.5) {
                xy = prev_outside_point;
                coords.y = -dot(xy - input.point_start, segment_right);
            } else if (input.position.y * sign_turn_right_start > 0.0) {
                // Outside corner of turn
                let tan_half_turn_angle = (1.0 - cos_turn_angle_start) / sin_turn_angle_start;
                let d = sign_turn_right_start * halfwidth * tan_half_turn_angle;
                xy = input.point_start - segment_right * (halfwidth * input.position.y) - d * segment_along;
            } else {
                xy = inside_point;
            }
        }
    } else {
        xy = point - segment_right * (halfwidth * input.position.y);
    }

    coords.x = dot(xy - input.point_start, segment_along);

    // Pack flags
    let turn_right_start = sin_turn_angle_start >= 0.0;
    let turn_right_end = sin_turn_angle_end >= 0.0;
    var flags = f32(i32(has_start_cap) +
                   2 * i32(has_end_cap) +
                   4 * i32(miter_too_large_start) +
                   8 * i32(miter_too_large_end) +
                   16 * i32(turn_right_start) +
                   32 * i32(turn_right_end));

    // Convert to clip space
    let pos = (xy + 0.5) / vec2f(uniforms.canvas_width, uniforms.canvas_height);
    output.position = vec4f(2.0 * pos.x - 1.0, 1.0 - 2.0 * pos.y, 0.0, 1.0);
    output.coords = coords;
    output.line_color = color;
    output.line_params = vec4f(linewidth, f32(cap_type), f32(join_type), segment_length);
    output.turn_angles = vec2f(cos_turn_angle_start, cos_turn_angle_end);
    output.flags = flags;

    return output;
}

fn turn_angle_to_right_vector(cos_turn_angle: f32, sign_turn_right: f32) -> vec2f {
    let sin_turn_angle = sign_turn_right * sqrt(1.0 - cos_turn_angle * cos_turn_angle);
    return vec2f(sin_turn_angle, -cos_turn_angle);
}

fn bevel_join_distance(coords: vec2f, other_right: vec2f, sign_turn_right: f32, linewidth: f32) -> f32 {
    let hw = 0.5 * linewidth;
    if (other_right.y >= ONE_MINUS_SMALL) {
        return abs(hw - coords.x);
    } else {
        let segment_right = vec2f(0.0, -1.0);
        let corner_right = normalize(other_right + segment_right);
        let outside_point = (-hw * sign_turn_right) * segment_right;
        return hw + sign_turn_right * dot(outside_point - coords, corner_right);
    }
}

fn cap_distance(cap_type: i32, x: f32, y: f32, linewidth: f32) -> f32 {
    let hw = 0.5 * linewidth;
    if (cap_type == CAP_BUTT) {
        return max(hw - x, abs(y));
    } else if (cap_type == CAP_SQUARE) {
        return max(-x, abs(y));
    } else {
        // Round cap
        return distance(vec2f(min(x, 0.0), y), vec2f(0.0, 0.0));
    }
}

fn distance_to_alpha(dist: f32, linewidth: f32, antialias: f32) -> f32 {
    return 1.0 - smoothstep(0.5 * (linewidth - antialias),
                            0.5 * (linewidth + antialias), dist);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let linewidth = input.line_params.x;
    let cap_type = i32(input.line_params.y + 0.5);
    let join_type = i32(input.line_params.z + 0.5);
    let segment_length = input.line_params.w;

    let halfwidth = 0.5 * (linewidth + uniforms.antialias);
    let half_antialias = 0.5 * uniforms.antialias;

    // Extract flags
    var flags = i32(input.flags + 0.5);
    let turn_right_end = (flags / 32) > 0;
    let sign_turn_right_end = select(-1.0, 1.0, turn_right_end);
    flags -= 32 * i32(turn_right_end);
    let turn_right_start = (flags / 16) > 0;
    let sign_turn_right_start = select(-1.0, 1.0, turn_right_start);
    flags -= 16 * i32(turn_right_start);
    let miter_too_large_end = (flags / 8) > 0;
    flags -= 8 * i32(miter_too_large_end);
    let miter_too_large_start = (flags / 4) > 0;
    flags -= 4 * i32(miter_too_large_start);
    let has_end_cap = (flags / 2) > 0;
    flags -= 2 * i32(has_end_cap);
    let has_start_cap = flags > 0;

    let prev_right = turn_angle_to_right_vector(input.turn_angles.x, sign_turn_right_start);
    let next_right = turn_angle_to_right_vector(input.turn_angles.y, sign_turn_right_end);

    var dist = input.coords.y;

    let end_coords = vec2f(segment_length, 0.0) - input.coords;

    // Handle start of segment
    if (input.coords.x <= half_antialias) {
        if (has_start_cap) {
            dist = cap_distance(cap_type, input.coords.x, input.coords.y, linewidth);
        } else if (join_type == JOIN_ROUND) {
            if (input.coords.x <= 0.0) {
                dist = distance(input.coords, vec2f(0.0, 0.0));
            }
        } else {
            if (join_type == JOIN_BEVEL || miter_too_large_start) {
                dist = max(abs(dist), bevel_join_distance(input.coords, prev_right, sign_turn_right_start, linewidth));
            }
            let prev_sideways_dist = -sign_turn_right_start * dot(input.coords, prev_right);
            dist = max(abs(dist), prev_sideways_dist);
        }
    }

    // Handle end of segment
    if (end_coords.x <= half_antialias) {
        if (has_end_cap) {
            dist = max(abs(dist), cap_distance(cap_type, end_coords.x, input.coords.y, linewidth));
        } else if (join_type == JOIN_BEVEL || miter_too_large_end) {
            dist = max(abs(dist), bevel_join_distance(end_coords, next_right, sign_turn_right_end, linewidth));
        }
    }

    let alpha = distance_to_alpha(abs(dist), linewidth, uniforms.antialias) * input.line_color.a;

    // Premultiplied alpha
    return vec4f(input.line_color.rgb * alpha, alpha);
}
