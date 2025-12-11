// WebGPU implementation for HexTile glyph
import type {Transform} from "./types"
import {BaseGPUGlyph} from "./base"
import type {WebGPUWrapper} from "./webgpu_wrapper"
import type {HexTileView} from "../hex_tile"
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
const UNIFORM_FLOATS = 8

export class HexTileGPU extends BaseGPUGlyph {
  private readonly _antialias: number = 1.5

  // GPU buffers
  private _position_buffer: GPUBuffer | null = null
  private _scale_buffer: GPUBuffer | null = null
  private _line_props_buffer: GPUBuffer | null = null
  private _line_color_buffer: GPUBuffer | null = null
  private _fill_color_buffer: GPUBuffer | null = null
  private _uniform_buffer: GPUBuffer | null = null
  private _bind_group: GPUBindGroup | null = null
  private _pipeline: GPURenderPipeline | null = null

  // CPU-side arrays
  private _position_data: Float32Array | null = null
  private _scale_data: Float32Array | null = null
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
  private _scale_upload_needed: boolean = true
  private _line_props_upload_needed: boolean = true
  private _line_color_upload_needed: boolean = true
  private _fill_color_upload_needed: boolean = true

  constructor(wrapper: WebGPUWrapper, override readonly glyph: HexTileView) {
    super(wrapper, glyph)
  }

  draw(indices: number[], main_glyph: GlyphView, transform: Transform): void {
    const main_hex_tile = main_glyph as HexTileView
    const main_gpu = main_hex_tile.gpuglyph

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

    this._draw_hex_tiles(indices, transform, main_gpu)
  }

  private _draw_hex_tiles(_indices: number[], transform: Transform, main_gpu: HexTileGPU): void {
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
      main_gpu._scale_upload_needed = true
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
    if (main_gpu._scale_upload_needed) {
      this._upload_buffer(device, main_gpu._scale_data!, main_gpu, "_scale_buffer")
      main_gpu._scale_upload_needed = false
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
    render_pass.setVertexBuffer(2, main_gpu._scale_buffer)
    render_pass.setVertexBuffer(3, main_gpu._line_props_buffer)
    render_pass.setVertexBuffer(4, main_gpu._line_color_buffer)
    render_pass.setVertexBuffer(5, main_gpu._fill_color_buffer)

    const scissor = webgpu_wrapper.scissor
    render_pass.setScissorRect(scissor.x, scissor.y, scissor.width, scissor.height)

    render_pass.draw(4, nmarkers, 0, 0)
    render_pass.end()

    device.queue.submit([command_encoder.finish()])
    webgpu_wrapper.mark_rendered()
  }

  private _update_uniforms(transform: Transform): void {
    const hex_tile = this.glyph
    const {svx, svy} = hex_tile
    const orientation = hex_tile.model.orientation

    // Compute hex dimensions from svx/svy
    // For flattop: width = 2*svx[0], height from svy
    // For pointytop: width = 2*svy[0], height from svx
    let hex_width: number
    let hex_height: number
    let rotation: number

    if (orientation === "pointytop") {
      // For pointytop, the hex is rotated by PI/2
      // Width comes from svy[0], height from svx
      hex_width = Math.abs(svy[0]) * 2
      hex_height = Math.abs(svx[4]) * 4 / Math.sqrt(3)
      rotation = Math.PI / 2
    } else {
      // Flattop orientation
      hex_width = Math.abs(svx[0]) * 2
      hex_height = Math.abs(svy[4]) * 4 / Math.sqrt(3)
      rotation = 0
    }

    this._uniform_data[0] = transform.width
    this._uniform_data[1] = transform.height
    this._uniform_data[2] = this._antialias / transform.pixel_ratio
    this._uniform_data[3] = 0 // padding
    this._uniform_data[4] = hex_width
    this._uniform_data[5] = hex_height
    this._uniform_data[6] = rotation
    this._uniform_data[7] = 0 // padding
  }

  private _ensure_pipeline(device: GPUDevice, wrapper: WebGPUWrapper): void {
    if (this._uniform_buffer == null) {
      this._uniform_buffer = device.createBuffer({
        size: UNIFORM_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
    }

    if (this._pipeline == null) {
      this._pipeline = wrapper.get_hex_tile_pipeline()
    }

    if (this._bind_group == null) {
      this._bind_group = wrapper.create_hex_tile_bind_group(this._uniform_buffer)
    }
  }

  private _upload_buffer(
    device: GPUDevice,
    data: Float32Array,
    target: HexTileGPU,
    buffer_name: "_position_buffer" | "_scale_buffer" | "_line_props_buffer" | "_line_color_buffer" | "_fill_color_buffer",
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
    this._scale_data = this._ensure_array(this._scale_data, nmarkers)

    const pos_data = this._position_data
    const scale_data = this._scale_data

    // HexTile uses sx, sy directly (screen coordinates)
    const {sx, sy, scale} = this.glyph

    for (let i = 0; i < nmarkers; i++) {
      const sx_i = sx[i]
      const sy_i = sy[i]

      if (!isFinite(sx_i) || !isFinite(sy_i)) {
        pos_data[i * 2] = MISSING_POINT
        pos_data[i * 2 + 1] = MISSING_POINT
      } else {
        pos_data[i * 2] = sx_i
        pos_data[i * 2 + 1] = sy_i
      }

      scale_data[i] = scale.get(i)
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
    this._scale_buffer?.destroy()
    this._line_props_buffer?.destroy()
    this._line_color_buffer?.destroy()
    this._fill_color_buffer?.destroy()
    this._uniform_buffer?.destroy()
  }
}
