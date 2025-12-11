// WebGPU implementation for MultiLine glyph
import type {Transform} from "./types"
import {BaseGPUGlyph} from "./base"
import type {WebGPUWrapper} from "./webgpu_wrapper"
import type {MultiLineView} from "../multi_line"
import type {GlyphView} from "../glyph"
import {color2rgba} from "core/util/color"

// Cap type enum values
const CAP_BUTT = 0
const CAP_ROUND = 1
const CAP_SQUARE = 2

// Join type enum values
const JOIN_MITER = 0
const JOIN_ROUND = 1
const JOIN_BEVEL = 2

const cap_lookup: Record<string, number> = {
  butt: CAP_BUTT,
  round: CAP_ROUND,
  square: CAP_SQUARE,
}

const join_lookup: Record<string, number> = {
  miter: JOIN_MITER,
  round: JOIN_ROUND,
  bevel: JOIN_BEVEL,
}

// Uniform buffer layout
const UNIFORM_FLOATS = 4  // canvas_width, canvas_height, antialias, miter_limit

export class MultiLineGPU extends BaseGPUGlyph {
  private readonly _antialias: number = 1.5
  private readonly _miter_limit: number = 10.0

  // GPU buffers - per line
  private _vertex_buffer: GPUBuffer | null = null
  private _index_buffer: GPUBuffer | null = null
  private _point_prev_buffer: GPUBuffer | null = null
  private _point_start_buffer: GPUBuffer | null = null
  private _point_end_buffer: GPUBuffer | null = null
  private _point_next_buffer: GPUBuffer | null = null
  private _show_flags_buffer: GPUBuffer | null = null
  private _line_props_buffer: GPUBuffer | null = null
  private _line_color_buffer: GPUBuffer | null = null
  private _uniform_buffer: GPUBuffer | null = null
  private _bind_group: GPUBindGroup | null = null
  private _pipeline: GPURenderPipeline | null = null

  // CPU-side arrays - pooled for all lines
  private _point_prev_data: Float32Array | null = null
  private _point_start_data: Float32Array | null = null
  private _point_end_data: Float32Array | null = null
  private _point_next_data: Float32Array | null = null
  private _show_flags_data: Float32Array | null = null
  private _line_props_data: Float32Array | null = null
  private _line_color_data: Float32Array | null = null
  private _uniform_data: Float32Array = new Float32Array(UNIFORM_FLOATS)

  // Track state
  private _data_dirty: boolean = true
  private _visuals_dirty: boolean = true
  private _last_total_segments: number = 0

  // Upload flags
  private _upload_needed: boolean = true

  constructor(wrapper: WebGPUWrapper, override readonly glyph: MultiLineView) {
    super(wrapper, glyph)
  }

  draw(indices: number[], main_glyph: GlyphView, transform: Transform): void {
    const main_multi = main_glyph as MultiLineView
    const main_gpu = main_multi.gpuglyph

    if (main_gpu == null) {
      return
    }

    if (main_gpu.data_changed) {
      main_gpu._data_dirty = true
      main_gpu.data_changed = false
    }

    if (main_gpu.data_mapped) {
      main_gpu._data_dirty = true
      main_gpu.data_mapped = false
    }

    if (this.visuals_changed) {
      main_gpu._visuals_dirty = true
      this.visuals_changed = false
    }

    this._draw_multi_line(indices, transform, main_gpu)
  }

  private _draw_multi_line(indices: number[], transform: Transform, main_gpu: MultiLineGPU): void {
    const {webgpu_wrapper} = this
    const device = webgpu_wrapper.device

    const nlines = main_gpu.glyph.data_size
    if (nlines === 0) {
      return
    }

    // Count total segments
    let total_segments = 0
    for (let i = 0; i < nlines; i++) {
      const npoints = main_gpu.glyph.sxs.get(i).length
      if (npoints > 1) {
        total_segments += npoints - 1
      }
    }

    if (total_segments === 0) {
      return
    }

    const size_changed = main_gpu._last_total_segments !== total_segments
    main_gpu._last_total_segments = total_segments

    // Update data buffers
    if (main_gpu._data_dirty || size_changed) {
      main_gpu._update_data(total_segments)
      main_gpu._data_dirty = false
      main_gpu._upload_needed = true
    }

    // Update visual buffers
    if (main_gpu._visuals_dirty || size_changed) {
      main_gpu._update_visuals(total_segments)
      main_gpu._visuals_dirty = false
      main_gpu._upload_needed = true
    }

    // Upload buffers
    if (main_gpu._upload_needed) {
      this._upload_all_buffers(device, main_gpu)
      main_gpu._upload_needed = false
    }

    // Update uniforms
    this._update_uniforms(transform)

    // Ensure pipeline exists
    this._ensure_pipeline(device, webgpu_wrapper)

    device.queue.writeBuffer(this._uniform_buffer!, 0, this._uniform_data.buffer, this._uniform_data.byteOffset, this._uniform_data.byteLength)

    // Render all lines
    const texture_view = webgpu_wrapper.context.getCurrentTexture().createView()
    const command_encoder = device.createCommandEncoder()
    const loadOp = webgpu_wrapper.get_load_op()

    const render_pass = command_encoder.beginRenderPass({
      colorAttachments: [{
        view: texture_view,
        clearValue: {r: 0, g: 0, b: 0, a: 0},
        loadOp,
        storeOp: "store",
      }],
    })

    render_pass.setPipeline(this._pipeline!)
    render_pass.setBindGroup(0, this._bind_group)
    render_pass.setVertexBuffer(0, this._vertex_buffer)
    render_pass.setVertexBuffer(1, main_gpu._point_prev_buffer)
    render_pass.setVertexBuffer(2, main_gpu._point_start_buffer)
    render_pass.setVertexBuffer(3, main_gpu._point_end_buffer)
    render_pass.setVertexBuffer(4, main_gpu._point_next_buffer)
    render_pass.setVertexBuffer(5, main_gpu._show_flags_buffer)
    render_pass.setVertexBuffer(6, main_gpu._line_props_buffer)
    render_pass.setVertexBuffer(7, main_gpu._line_color_buffer)

    const scissor = webgpu_wrapper.scissor
    render_pass.setScissorRect(scissor.x, scissor.y, scissor.width, scissor.height)

    // Set index buffer for indexed drawing
    render_pass.setIndexBuffer(this._index_buffer!, "uint32")

    // Draw only segments for selected indices
    if (indices.length === main_gpu.glyph.data_size) {
      // All lines selected, draw everything
      render_pass.drawIndexed(9, total_segments, 0, 0, 0)
    } else {
      // Draw segments for each selected line
      let segment_offset = 0
      let prev_line_idx = -1

      for (const line_idx of indices) {
        // Skip segments for unselected lines
        for (let i = prev_line_idx + 1; i < line_idx; i++) {
          const npoints = main_gpu.glyph.sxs.get(i).length
          if (npoints > 1) {
            segment_offset += npoints - 1
          }
        }

        const npoints = main_gpu.glyph.sxs.get(line_idx).length
        const nsegments = npoints > 1 ? npoints - 1 : 0

        if (nsegments > 0) {
          render_pass.drawIndexed(9, nsegments, 0, 0, segment_offset)
          segment_offset += nsegments
        }

        prev_line_idx = line_idx
      }
    }

    render_pass.end()

    device.queue.submit([command_encoder.finish()])
    webgpu_wrapper.mark_rendered()
  }

  private _update_uniforms(transform: Transform): void {
    this._uniform_data[0] = transform.width
    this._uniform_data[1] = transform.height
    this._uniform_data[2] = this._antialias / transform.pixel_ratio
    this._uniform_data[3] = this._miter_limit
  }

  private _ensure_pipeline(device: GPUDevice, wrapper: WebGPUWrapper): void {
    // Create vertex buffer for geometry (5 vertices like WebGL)
    if (this._vertex_buffer == null) {
      const vertices = new Float32Array([
        -2,  0,  // vertex 0: join point (center y)
        -1, -1,  // vertex 1: start bottom
         1, -1,  // vertex 2: end bottom
         1,  1,  // vertex 3: end top
        -1,  1,  // vertex 4: start top
      ])
      this._vertex_buffer = device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(this._vertex_buffer, 0, vertices)
    }

    // Create index buffer for triangles
    if (this._index_buffer == null) {
      const indices = new Uint32Array([
        0, 1, 4,
        1, 2, 4,
        2, 4, 3,
      ])
      this._index_buffer = device.createBuffer({
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(this._index_buffer, 0, indices)
    }

    if (this._uniform_buffer == null) {
      this._uniform_buffer = device.createBuffer({
        size: UNIFORM_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
    }

    if (this._pipeline == null) {
      this._pipeline = wrapper.get_line_pipeline()
    }

    if (this._bind_group == null) {
      this._bind_group = wrapper.create_line_bind_group(this._uniform_buffer)
    }
  }

  private _upload_all_buffers(device: GPUDevice, target: MultiLineGPU): void {
    this._upload_buffer(device, target._point_prev_data!, target, "_point_prev_buffer")
    this._upload_buffer(device, target._point_start_data!, target, "_point_start_buffer")
    this._upload_buffer(device, target._point_end_data!, target, "_point_end_buffer")
    this._upload_buffer(device, target._point_next_data!, target, "_point_next_buffer")
    this._upload_buffer(device, target._show_flags_data!, target, "_show_flags_buffer")
    this._upload_buffer(device, target._line_props_data!, target, "_line_props_buffer")
    this._upload_buffer(device, target._line_color_data!, target, "_line_color_buffer")
  }

  private _upload_buffer(
    device: GPUDevice,
    data: Float32Array,
    target: MultiLineGPU,
    buffer_name: "_point_prev_buffer" | "_point_start_buffer" | "_point_end_buffer" | "_point_next_buffer" | "_show_flags_buffer" | "_line_props_buffer" | "_line_color_buffer",
  ): void {
    const byte_size = data.byteLength
    const existing = target[buffer_name]

    if (existing == null || existing.size < byte_size) {
      existing?.destroy()
      target[buffer_name] = device.createBuffer({
        size: byte_size,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
    }
    device.queue.writeBuffer(target[buffer_name]!, 0, data.buffer, data.byteOffset, data.byteLength)
  }

  private _ensure_array(current: Float32Array | null, required_length: number): Float32Array {
    if (current == null || current.length < required_length) {
      return new Float32Array(required_length)
    }
    return current
  }

  private _update_data(total_segments: number): void {
    this._point_prev_data = this._ensure_array(this._point_prev_data, total_segments * 2)
    this._point_start_data = this._ensure_array(this._point_start_data, total_segments * 2)
    this._point_end_data = this._ensure_array(this._point_end_data, total_segments * 2)
    this._point_next_data = this._ensure_array(this._point_next_data, total_segments * 2)
    this._show_flags_data = this._ensure_array(this._show_flags_data, total_segments * 4)

    const {sxs, sys} = this.glyph
    const nlines = this.glyph.data_size

    let seg_idx = 0
    for (let line = 0; line < nlines; line++) {
      const sx = sxs.get(line)
      const sy = sys.get(line)
      const npoints = sx.length
      const nsegments = npoints > 1 ? npoints - 1 : 0

      if (nsegments === 0) {
        continue
      }

      // Check if line is closed
      const is_closed = npoints > 2 &&
                        sx[0] === sx[npoints - 1] &&
                        sy[0] === sy[npoints - 1] &&
                        isFinite(sx[0] + sy[0])

      // Pre-compute finite flags
      const finite: boolean[] = new Array(npoints)
      for (let i = 0; i < npoints; i++) {
        finite[i] = isFinite(sx[i] + sy[i])
      }

      for (let seg = 0; seg < nsegments; seg++) {
        const i = seg
        const j = seg + 1

        // Start and end points
        this._point_start_data[seg_idx * 2] = sx[i]
        this._point_start_data[seg_idx * 2 + 1] = sy[i]
        this._point_end_data[seg_idx * 2] = sx[j]
        this._point_end_data[seg_idx * 2 + 1] = sy[j]

        // Previous point
        let prev_idx: number
        if (i > 0) {
          prev_idx = i - 1
        } else if (is_closed) {
          prev_idx = npoints - 2
        } else {
          prev_idx = i
        }
        this._point_prev_data[seg_idx * 2] = sx[prev_idx]
        this._point_prev_data[seg_idx * 2 + 1] = sy[prev_idx]

        // Next point
        let next_idx: number
        if (j < npoints - 1) {
          next_idx = j + 1
        } else if (is_closed) {
          next_idx = 1
        } else {
          next_idx = j
        }
        this._point_next_data[seg_idx * 2] = sx[next_idx]
        this._point_next_data[seg_idx * 2 + 1] = sy[next_idx]

        // Show flags
        const start_finite = finite[i]
        const end_finite = finite[j]
        const show_curr = start_finite && end_finite

        let show_prev: boolean
        if (i === 0 && !is_closed) {
          show_prev = false
        } else {
          show_prev = finite[prev_idx] && start_finite
        }

        let show_next: boolean
        if (j === npoints - 1 && !is_closed) {
          show_next = false
        } else {
          show_next = end_finite && finite[next_idx]
        }

        this._show_flags_data[seg_idx * 4] = show_prev ? 1 : 0
        this._show_flags_data[seg_idx * 4 + 1] = show_curr ? 1 : 0
        this._show_flags_data[seg_idx * 4 + 2] = show_next ? 1 : 0
        this._show_flags_data[seg_idx * 4 + 3] = 0

        seg_idx++
      }
    }
  }

  private _update_visuals(total_segments: number): void {
    this._line_props_data = this._ensure_array(this._line_props_data, total_segments * 4)
    this._line_color_data = this._ensure_array(this._line_color_data, total_segments * 4)

    const visuals = this.glyph.visuals
    const line = visuals.line

    const {sxs} = this.glyph
    const nlines = this.glyph.data_size

    let seg_idx = 0
    for (let line_idx = 0; line_idx < nlines; line_idx++) {
      const npoints = sxs.get(line_idx).length
      const nsegments = npoints > 1 ? npoints - 1 : 0

      if (nsegments === 0) {
        continue
      }

      // Get per-line visual properties (LineVector)
      const linewidth = line.line_width.get(line_idx)
      const cap = cap_lookup[line.line_cap.get(line_idx)] ?? CAP_BUTT
      const join = join_lookup[line.line_join.get(line_idx)] ?? JOIN_MITER

      const color = line.line_color.get(line_idx)
      const alpha = line.line_alpha.get(line_idx)
      // Use color2rgba to properly handle both string colors and uint32 values
      const [r8, g8, b8, a8] = color2rgba(color, alpha)
      const r = r8 / 255
      const g = g8 / 255
      const b = b8 / 255
      const final_alpha = a8 / 255

      // Apply to all segments of this line
      for (let seg = 0; seg < nsegments; seg++) {
        this._line_props_data[seg_idx * 4] = linewidth
        this._line_props_data[seg_idx * 4 + 1] = cap
        this._line_props_data[seg_idx * 4 + 2] = join
        this._line_props_data[seg_idx * 4 + 3] = 0

        this._line_color_data[seg_idx * 4] = r
        this._line_color_data[seg_idx * 4 + 1] = g
        this._line_color_data[seg_idx * 4 + 2] = b
        this._line_color_data[seg_idx * 4 + 3] = final_alpha

        seg_idx++
      }
    }
  }

  destroy(): void {
    this._vertex_buffer?.destroy()
    this._index_buffer?.destroy()
    this._point_prev_buffer?.destroy()
    this._point_start_buffer?.destroy()
    this._point_end_buffer?.destroy()
    this._point_next_buffer?.destroy()
    this._show_flags_buffer?.destroy()
    this._line_props_buffer?.destroy()
    this._line_color_buffer?.destroy()
    this._uniform_buffer?.destroy()
  }
}
