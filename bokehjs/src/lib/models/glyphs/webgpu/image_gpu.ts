// WebGPU implementation for Image glyph
import type {Transform} from "./types"
import {BaseGPUGlyph} from "./base"
import type {WebGPUWrapper} from "./webgpu_wrapper"
import type {ImageBaseView} from "../image_base"
import type {GlyphView} from "../glyph"

// Uniform buffer layout (must match shader)
const UNIFORM_FLOATS = 8

export class ImageGPU extends BaseGPUGlyph {
  // GPU resources per image
  private _textures: (GPUTexture | null)[] = []
  private _bind_groups: (GPUBindGroup | null)[] = []

  // Uniform buffer (shared across images, updated per draw)
  private _uniform_buffer: GPUBuffer | null = null
  private _uniform_data: Float32Array = new Float32Array(UNIFORM_FLOATS)

  // Pipeline (shared)
  private _pipeline: GPURenderPipeline | null = null

  // Sampler (shared)
  private _sampler: GPUSampler | null = null

  // Image changed flag (separate from data_changed as it can occur through changed colormapping)
  private _image_changed: boolean = false

  constructor(wrapper: WebGPUWrapper, override readonly glyph: ImageBaseView) {
    super(wrapper, glyph)
  }

  draw(indices: number[], main_glyph: GlyphView, transform: Transform): void {
    const main_image = main_glyph as ImageBaseView
    const main_gpu = main_image.gpuglyph as ImageGPU | undefined

    if (main_gpu == null) {
      return
    }

    const data_changed_or_mapped = main_gpu.data_changed || main_gpu.data_mapped

    if (data_changed_or_mapped) {
      main_gpu._set_data()
    }

    // Always call _set_image if textures haven't been created yet (on main_gpu where data lives)
    if (main_gpu._image_changed || main_gpu.data_changed || main_gpu._textures.length === 0) {
      // Ensure pipeline is ready on THIS instance (need uniform buffer and sampler for bind groups)
      const device = this.webgpu_wrapper.device
      this._ensure_pipeline(device, this.webgpu_wrapper)

      // Now set images on main_gpu, passing this's resources
      main_gpu._set_image_with_resources(this._uniform_buffer!, this._sampler!, this.webgpu_wrapper)
    }

    main_gpu.data_changed = false
    main_gpu.data_mapped = false
    main_gpu._image_changed = false

    this._draw_images(indices, transform, main_gpu)
  }

  set_image_changed(): void {
    this._image_changed = true
  }

  private _draw_images(indices: number[], transform: Transform, main_gpu: ImageGPU): void {
    const {webgpu_wrapper} = this
    const device = webgpu_wrapper.device

    const {global_alpha} = this.glyph.visuals.image

    // Get the render target
    const texture_view = webgpu_wrapper.context.getCurrentTexture().createView()
    let loadOp = webgpu_wrapper.get_load_op()

    const scissor = webgpu_wrapper.scissor

    // Render each image - must submit individually because uniform buffer is shared
    // and writeBuffer + render commands are batched
    for (const i of indices) {
      if (main_gpu._textures[i] == null || main_gpu._bind_groups[i] == null) {
        continue
      }

      const bounds = main_gpu._get_bounds(i)
      if (bounds == null) {
        continue
      }

      // Update uniforms for this image
      this._uniform_data[0] = transform.width
      this._uniform_data[1] = transform.height
      this._uniform_data[2] = global_alpha.get(i)
      this._uniform_data[3] = 0 // padding
      this._uniform_data[4] = bounds[0] // x0
      this._uniform_data[5] = bounds[1] // y0
      this._uniform_data[6] = bounds[2] // x1
      this._uniform_data[7] = bounds[3] // y1

      device.queue.writeBuffer(this._uniform_buffer!, 0, this._uniform_data.buffer, this._uniform_data.byteOffset, this._uniform_data.byteLength)

      // Create a new command encoder for each image to ensure uniform buffer
      // has correct value when the render pass executes
      const command_encoder = device.createCommandEncoder()

      const render_pass = command_encoder.beginRenderPass({
        colorAttachments: [{
          view: texture_view,
          clearValue: {r: 0, g: 0, b: 0, a: 0},
          loadOp,
          storeOp: "store",
        }],
      })

      render_pass.setPipeline(this._pipeline!)
      render_pass.setBindGroup(0, main_gpu._bind_groups[i])
      render_pass.setVertexBuffer(0, webgpu_wrapper.rect_geometry)
      render_pass.setScissorRect(scissor.x, scissor.y, scissor.width, scissor.height)

      render_pass.draw(4, 1, 0, 0)
      render_pass.end()

      // Submit immediately to ensure uniform buffer has correct value
      device.queue.submit([command_encoder.finish()])

      // After first render, use "load" to preserve previous content
      loadOp = "load"
    }

    webgpu_wrapper.mark_rendered()
  }

  private _get_bounds(i: number): [number, number, number, number] | null {
    const {sx, sy, sdw, sdh, xy_anchor, xy_scale, xy_sign} = this.glyph

    const sx_i = sx[i]
    const sy_i = sy[i]
    const sw_i = sdw[i]
    const sh_i = sdh[i]

    if (!isFinite(sx_i + sy_i + sw_i + sh_i)) {
      return null
    }

    const x0 = sx_i + sw_i * (0.5 * (1 - xy_scale.x) - xy_anchor.x) * xy_sign.x
    const y0 = sy_i + sh_i * (0.5 * (1 - xy_scale.y) - xy_anchor.y) * xy_sign.y
    const x1 = x0 + sw_i * xy_scale.x * xy_sign.x
    const y1 = y0 + sh_i * xy_scale.y * xy_sign.y

    return [x0, y0, x1, y1]
  }

  private _ensure_pipeline(device: GPUDevice, wrapper: WebGPUWrapper): void {
    if (this._uniform_buffer == null) {
      this._uniform_buffer = device.createBuffer({
        size: UNIFORM_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
    }

    if (this._sampler == null) {
      this._sampler = device.createSampler({
        magFilter: "nearest",
        minFilter: "nearest",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
      })
    }

    if (this._pipeline == null) {
      this._pipeline = wrapper.get_image_pipeline()
    }
  }

  private _set_data(): void {
    // Data (bounds) are computed on-the-fly in _get_bounds
    // No pre-computation needed
  }

  // Called with resources from the drawing instance
  _set_image_with_resources(uniform_buffer: GPUBuffer, sampler: GPUSampler, wrapper: WebGPUWrapper): void {
    const {image_data} = this.glyph
    if (image_data == null || image_data.length === 0) {
      return
    }

    const device = wrapper.device
    const nimage = image_data.length

    // Resize arrays if needed
    if (this._textures.length !== nimage) {
      // Destroy old textures
      for (const tex of this._textures) {
        tex?.destroy()
      }
      this._textures = new Array(nimage).fill(null)
      this._bind_groups = new Array(nimage).fill(null)
    }

    for (let i = 0; i < nimage; i++) {
      const canvas = image_data[i]
      if (canvas == null) {
        this._textures[i] = null
        this._bind_groups[i] = null
        continue
      }

      const width = canvas.width
      const height = canvas.height

      // Create or recreate texture if size changed
      const existing_tex = this._textures[i]
      if (existing_tex == null || existing_tex.width !== width || existing_tex.height !== height) {
        existing_tex?.destroy()

        this._textures[i] = device.createTexture({
          size: {width, height},
          format: "rgba8unorm",
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        })

        // Create bind group for this texture using passed-in resources
        this._bind_groups[i] = wrapper.create_image_bind_group(
          uniform_buffer,
          this._textures[i]!.createView(),
          sampler,
        )
      }

      // Copy canvas data to texture
      // Get ImageData from the canvas
      const ctx = canvas.getContext("2d", {willReadFrequently: true})
      if (ctx == null) {
        continue
      }

      const imageData = ctx.getImageData(0, 0, width, height)

      device.queue.writeTexture(
        {texture: this._textures[i]!},
        imageData.data,
        {bytesPerRow: width * 4, rowsPerImage: height},
        {width, height},
      )
    }
  }

  destroy(): void {
    for (const tex of this._textures) {
      tex?.destroy()
    }
    this._textures = []
    this._bind_groups = []
    this._uniform_buffer?.destroy()
  }
}
