// WebGPU implementation for Ngon glyph
import type {Transform} from "./types"
import {BaseGPUGlyph} from "./base"
import type {WebGPUWrapper} from "./webgpu_wrapper"
import type {NgonView} from "../ngon"
import type {GlyphView} from "../glyph"

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

// Missing data marker value
const MISSING_POINT = -10000

// Uniform buffer layout (must match shader)
const UNIFORM_FLOATS = 12

export class NgonGPU extends BaseGPUGlyph {
  private readonly _antialias: number = 1.5

  // GPU buffers
  private _position_buffer: GPUBuffer | null = null
  private _radius_buffer: GPUBuffer | null = null
  private _geometry_buffer: GPUBuffer | null = null
  private _line_props_buffer: GPUBuffer | null = null
  private _line_color_buffer: GPUBuffer | null = null
  private _fill_color_buffer: GPUBuffer | null = null
  private _uniform_buffer: GPUBuffer | null = null
  private _bind_group: GPUBindGroup | null = null
  private _pipeline: GPURenderPipeline | null = null

  // CPU-side arrays
  private _position_data: Float32Array | null = null
  private _radius_data: Float32Array | null = null
  private _geometry_data: Float32Array | null = null
  private _line_props_data: Float32Array | null = null
  private _line_color_data: Float32Array | null = null
  private _fill_color_data: Float32Array | null = null
  private _uniform_data: Float32Array = new Float32Array(UNIFORM_FLOATS)

  // Track state
  private _data_dirty: boolean = true
  private _visuals_dirty: boolean = true
  private _last_nmarkers: number = 0

  // Upload flags
  private _positions_upload_needed: boolean = true
  private _radius_upload_needed: boolean = true
  private _geometry_upload_needed: boolean = true
  private _line_props_upload_needed: boolean = true
  private _line_color_upload_needed: boolean = true
  private _fill_color_upload_needed: boolean = true

  constructor(wrapper: WebGPUWrapper, override readonly glyph: NgonView) {
    super(wrapper, glyph)
  }

  draw(indices: number[], main_glyph: GlyphView, transform: Transform): void {
    const main_ngon = main_glyph as NgonView
    const main_gpu = main_ngon.gpuglyph

    if (main_gpu == null) {
      return
    }

    if (main_gpu.data_changed) {
      main_gpu._data_dirty = true
      main_gpu.data_changed = false
    }

    if (main_gpu.data_mapped) {
      main_gpu.data_mapped = false
    }

    if (this.visuals_changed) {
      main_gpu._visuals_dirty = true
      this.visuals_changed = false
    }

    this._draw_ngons(indices, transform, main_gpu)
  }

  private _draw_ngons(_indices: number[], transform: Transform, main_gpu: NgonGPU): void {
    const {webgpu_wrapper} = this
    const device = webgpu_wrapper.device

    // Use glyph's data_size directly instead of cached nvertices
    const nmarkers = main_gpu.glyph.data_size
    if (nmarkers === 0) {
      return
    }

    if (main_gpu._position_data == null && !main_gpu._data_dirty) {
      main_gpu._data_dirty = true
    }

    const size_changed = main_gpu._last_nmarkers !== nmarkers
    main_gpu._last_nmarkers = nmarkers

    // Update data buffers
    if (main_gpu._data_dirty || size_changed) {
      main_gpu._update_data(nmarkers)
      main_gpu._data_dirty = false
      main_gpu._positions_upload_needed = true
      main_gpu._radius_upload_needed = true
      main_gpu._geometry_upload_needed = true
    }

    // Update visual buffers
    if (main_gpu._visuals_dirty || size_changed) {
      main_gpu._update_visuals(nmarkers)
      main_gpu._visuals_dirty = false
      main_gpu._line_props_upload_needed = true
      main_gpu._line_color_upload_needed = true
      main_gpu._fill_color_upload_needed = true
    }

    // Set show flags
    if (size_changed) {
      main_gpu._update_show_flags_all(nmarkers)
      main_gpu._line_props_upload_needed = true
    }

    // Upload buffers
    if (main_gpu._positions_upload_needed) {
      this._upload_buffer(device, main_gpu._position_data!, main_gpu, "_position_buffer")
      main_gpu._positions_upload_needed = false
    }
    if (main_gpu._radius_upload_needed) {
      this._upload_buffer(device, main_gpu._radius_data!, main_gpu, "_radius_buffer")
      main_gpu._radius_upload_needed = false
    }
    if (main_gpu._geometry_upload_needed) {
      this._upload_buffer(device, main_gpu._geometry_data!, main_gpu, "_geometry_buffer")
      main_gpu._geometry_upload_needed = false
    }
    if (main_gpu._line_props_upload_needed) {
      this._upload_buffer(device, main_gpu._line_props_data!, main_gpu, "_line_props_buffer")
      main_gpu._line_props_upload_needed = false
    }
    if (main_gpu._line_color_upload_needed) {
      this._upload_buffer(device, main_gpu._line_color_data!, main_gpu, "_line_color_buffer")
      main_gpu._line_color_upload_needed = false
    }
    if (main_gpu._fill_color_upload_needed) {
      this._upload_buffer(device, main_gpu._fill_color_data!, main_gpu, "_fill_color_buffer")
      main_gpu._fill_color_upload_needed = false
    }

    // Update uniforms
    this._update_uniforms(transform)

    // Ensure pipeline and bind group
    this._ensure_pipeline(device, webgpu_wrapper)

    device.queue.writeBuffer(this._uniform_buffer!, 0, this._uniform_data.buffer, this._uniform_data.byteOffset, this._uniform_data.byteLength)

    // Render
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
    render_pass.setVertexBuffer(0, webgpu_wrapper.rect_geometry)
    render_pass.setVertexBuffer(1, main_gpu._position_buffer)
    render_pass.setVertexBuffer(2, main_gpu._radius_buffer)
    render_pass.setVertexBuffer(3, main_gpu._geometry_buffer)
    render_pass.setVertexBuffer(4, main_gpu._line_props_buffer)
    render_pass.setVertexBuffer(5, main_gpu._line_color_buffer)
    render_pass.setVertexBuffer(6, main_gpu._fill_color_buffer)

    const scissor = webgpu_wrapper.scissor
    render_pass.setScissorRect(scissor.x, scissor.y, scissor.width, scissor.height)

    render_pass.draw(4, nmarkers, 0, 0)
    render_pass.end()

    device.queue.submit([command_encoder.finish()])
    webgpu_wrapper.mark_rendered()
  }

  private _update_uniforms(transform: Transform): void {
    const renderer = this.glyph.renderer
    const frame = renderer.plot_view.frame

    const x_scale = frame.x_scales.get(renderer.model.x_range_name)
    const y_scale = frame.y_scales.get(renderer.model.y_range_name)

    if (x_scale == null || y_scale == null) {
      return
    }

    const x_source = x_scale.source_range
    const x_target = x_scale.target_range
    const y_source = y_scale.source_range
    const y_target = y_scale.target_range

    const x_factor = (x_target.end - x_target.start) / (x_source.end - x_source.start)
    const x_offset = -(x_factor * x_source.start) + x_target.start

    const y_factor = (y_target.end - y_target.start) / (y_source.end - y_source.start)
    const y_offset = -(y_factor * y_source.start) + y_target.start

    const radius_scale = Math.abs(x_factor)

    this._uniform_data[0] = transform.width
    this._uniform_data[1] = transform.height
    this._uniform_data[2] = this._antialias / transform.pixel_ratio
    this._uniform_data[3] = 0 // padding
    this._uniform_data[4] = x_factor
    this._uniform_data[5] = x_offset
    this._uniform_data[6] = y_factor
    this._uniform_data[7] = y_offset
    this._uniform_data[8] = radius_scale
    this._uniform_data[9] = 0
    this._uniform_data[10] = 0
    this._uniform_data[11] = 0
  }

  private _ensure_pipeline(device: GPUDevice, wrapper: WebGPUWrapper): void {
    if (this._uniform_buffer == null) {
      this._uniform_buffer = device.createBuffer({
        size: UNIFORM_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
    }

    if (this._pipeline == null) {
      this._pipeline = wrapper.get_ngon_pipeline()
    }

    if (this._bind_group == null) {
      this._bind_group = wrapper.create_ngon_bind_group(this._uniform_buffer)
    }
  }

  private _upload_buffer(
    device: GPUDevice,
    data: Float32Array,
    target: NgonGPU,
    buffer_name: "_position_buffer" | "_radius_buffer" | "_geometry_buffer" | "_line_props_buffer" | "_line_color_buffer" | "_fill_color_buffer",
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

  private _update_data(nmarkers: number): void {
    this._position_data = this._ensure_array(this._position_data, nmarkers * 2)
    this._radius_data = this._ensure_array(this._radius_data, nmarkers)
    this._geometry_data = this._ensure_array(this._geometry_data, nmarkers * 2)

    const pos_data = this._position_data
    const radius_data = this._radius_data
    const geom_data = this._geometry_data

    const {x, y, radius, angle, n} = this.glyph

    for (let i = 0; i < nmarkers; i++) {
      const x_i = x[i]
      const y_i = y[i]

      if (!isFinite(x_i) || !isFinite(y_i)) {
        pos_data[i * 2] = MISSING_POINT
        pos_data[i * 2 + 1] = MISSING_POINT
      } else {
        pos_data[i * 2] = x_i
        pos_data[i * 2 + 1] = y_i
      }

      radius_data[i] = radius.get(i)

      geom_data[i * 2] = angle.get(i)
      geom_data[i * 2 + 1] = n.get(i)
    }
  }

  private _update_visuals(nmarkers: number): void {
    this._line_props_data = this._ensure_array(this._line_props_data, nmarkers * 4)
    this._line_color_data = this._ensure_array(this._line_color_data, nmarkers * 4)
    this._fill_color_data = this._ensure_array(this._fill_color_data, nmarkers * 4)

    const line_props = this._line_props_data
    const line_color = this._line_color_data
    const fill_color = this._fill_color_data

    const visuals = this.glyph.visuals
    const line = visuals.line
    const fill = visuals.fill

    for (let i = 0; i < nmarkers; i++) {
      const offset = i * 4

      line_props[offset] = line.line_width.get(i)
      line_props[offset + 1] = cap_lookup[line.line_cap.get(i)] ?? CAP_BUTT
      line_props[offset + 2] = join_lookup[line.line_join.get(i)] ?? JOIN_MITER
      line_props[offset + 3] = 1.0

      const lc = line.line_color.get(i)
      const lc_a = (lc & 0xff) / 255
      const line_alpha = line.line_alpha.get(i) * lc_a
      line_color[offset] = ((lc >> 24) & 0xff) / 255
      line_color[offset + 1] = ((lc >> 16) & 0xff) / 255
      line_color[offset + 2] = ((lc >> 8) & 0xff) / 255
      line_color[offset + 3] = line_alpha

      const fc = fill.fill_color.get(i)
      const fc_a = (fc & 0xff) / 255
      const fill_alpha = fill.fill_alpha.get(i) * fc_a
      fill_color[offset] = ((fc >> 24) & 0xff) / 255
      fill_color[offset + 1] = ((fc >> 16) & 0xff) / 255
      fill_color[offset + 2] = ((fc >> 8) & 0xff) / 255
      fill_color[offset + 3] = fill_alpha
    }
  }

  private _update_show_flags_all(nmarkers: number): void {
    if (this._line_props_data == null) {
      return
    }

    const line_props = this._line_props_data
    for (let i = 0; i < nmarkers; i++) {
      line_props[i * 4 + 3] = 1.0
    }
  }

  destroy(): void {
    this._position_buffer?.destroy()
    this._radius_buffer?.destroy()
    this._geometry_buffer?.destroy()
    this._line_props_buffer?.destroy()
    this._line_color_buffer?.destroy()
    this._fill_color_buffer?.destroy()
    this._uniform_buffer?.destroy()
  }
}
