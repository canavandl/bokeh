import type {WebGPUWrapper} from "./webgpu_wrapper"
import type {LineCap, LineJoin} from "core/enums"
import type {HatchPattern} from "core/property_mixins"
import type {uint32, Arrayable} from "core/types"
import type {Uniform, ColorUniformVector} from "core/uniforms"
import {assert} from "core/util/assert"
import {color2rgba, byte} from "core/util/color"

// Lookup tables for enum values (same as WebGL)
export const cap_lookup: Record<LineCap, number> = {
  butt: 0,
  round: 1,
  square: 2,
}

export const join_lookup: Record<LineJoin, number> = {
  miter: 0,
  round: 1,
  bevel: 2,
}

// Hatch pattern to index mapping (same as WebGL)
const hatch_pattern_map: Record<HatchPattern, number> = {
  " ": 0, blank: 0,
  ".": 1, dot: 1,
  o: 2, ring: 2,
  "-": 3, horizontal_line: 3,
  "|": 4, vertical_line: 4,
  "+": 5, cross: 5,
  "\"": 6, horizontal_dash: 6,
  ":": 7, vertical_dash: 7,
  "@": 8, spiral: 8,
  "/": 9, right_diagonal_line: 9,
  "\\": 10, left_diagonal_line: 10,
  x: 11, diagonal_cross: 11,
  ",": 12, right_diagonal_dash: 12,
  "`": 13, left_diagonal_dash: 13,
  v: 14, horizontal_wave: 14,
  ">": 15, vertical_wave: 15,
  "*": 16, criss_cross: 16,
}

export function hatch_pattern_to_index(pattern: HatchPattern): number {
  return hatch_pattern_map[pattern] ?? 0
}

type WrappedArrayType = Float32Array | Uint8Array

// Base class for GPU buffers - manages both CPU-side array and GPU buffer
abstract class WrappedBuffer<ArrayType extends WrappedArrayType> {
  protected wrapper: WebGPUWrapper
  protected buffer: GPUBuffer | null = null
  protected array: ArrayType | null = null
  protected is_scalar: boolean = true
  protected elements_per_primitive: number

  constructor(wrapper: WebGPUWrapper, elements_per_primitive: number = 1) {
    this.wrapper = wrapper
    this.elements_per_primitive = elements_per_primitive
  }

  protected abstract bytes_per_element(): number
  protected abstract new_array(len: number): ArrayType

  // Return array if already exists
  get_array(): ArrayType {
    assert(this.array != null, "GPUBuffer not yet initialized")
    return this.array
  }

  // Return array of correct size, creating if necessary
  get_sized_array(length: number): ArrayType {
    if (this.array == null || this.array.length != length) {
      this.array = this.new_array(length)
    }
    return this.array
  }

  get length(): number {
    return this.array != null ? this.array.length : 0
  }

  set_from_array(numbers: Arrayable<number>): void {
    const len = numbers.length
    const array = this.get_sized_array(len)

    for (let i = 0; i < len; i++) {
      array[i] = numbers[i]
    }

    this.update()
  }

  set_from_prop(prop: Uniform<number>): void {
    const len = prop.is_Scalar() ? 1 : prop.length
    const array = this.get_sized_array(len)

    for (let i = 0; i < len; i++) {
      array[i] = prop.get(i)
    }

    this.update(prop.is_Scalar())
  }

  set_from_scalar(scalar: number): void {
    this.get_sized_array(1).fill(scalar)
    this.update(true)
  }

  // Update GPU buffer with CPU-side array data
  update(is_scalar: boolean = false): void {
    if (this.array == null) {
      return
    }

    const byte_size = this.array.byteLength

    // Create or resize buffer if needed
    if (this.buffer == null || this.buffer.size < byte_size) {
      this.buffer?.destroy()
      this.buffer = this.wrapper.device.createBuffer({
        size: byte_size,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
    }

    // Write data to GPU buffer
    this.wrapper.device.queue.writeBuffer(this.buffer, 0, this.array.buffer, this.array.byteOffset, this.array.byteLength)
    this.is_scalar = is_scalar
  }

  get gpu_buffer(): GPUBuffer {
    assert(this.buffer != null, "GPUBuffer not yet initialized")
    return this.buffer
  }

  destroy(): void {
    this.buffer?.destroy()
    this.buffer = null
  }
}

export class Float32Buffer extends WrappedBuffer<Float32Array> {
  protected bytes_per_element(): number {
    return Float32Array.BYTES_PER_ELEMENT
  }

  protected new_array(len: number): Float32Array {
    return new Float32Array(len)
  }
}

export class Uint8Buffer extends WrappedBuffer<Uint8Array> {
  protected bytes_per_element(): number {
    return Uint8Array.BYTES_PER_ELEMENT
  }

  protected new_array(len: number): Uint8Array {
    return new Uint8Array(len)
  }

  set_from_color(color_prop: Uniform<uint32>, alpha_prop: Uniform<number>): void {
    const is_scalar_colors = color_prop.is_Scalar()
    const is_scalar = is_scalar_colors && alpha_prop.is_Scalar()
    const ncolors = is_scalar ? 1 : color_prop.length

    if (!is_scalar_colors) {
      const color_v = color_prop as ColorUniformVector
      const array = new Uint8Array(color_v.copy_buffer())
      for (let i = 0; i < ncolors; i++) {
        const alpha = alpha_prop.get(i)
        array[4*i+3] = byte(alpha*array[4*i+3])
      }
      this.array = array
      this.update(is_scalar)
      return
    }

    const array = this.get_sized_array(4*ncolors)

    for (let i = 0; i < ncolors; i++) {
      const [r, g, b, a] = color2rgba(color_prop.get(i), alpha_prop.get(i))
      array[4*i  ] = r
      array[4*i+1] = g
      array[4*i+2] = b
      array[4*i+3] = a
    }

    this.update(is_scalar)
  }

  set_from_hatch_pattern(hatch_pattern_prop: Uniform<HatchPattern>): void {
    const len = hatch_pattern_prop.is_Scalar() ? 1 : hatch_pattern_prop.length
    const array = this.get_sized_array(len)

    for (let i = 0; i < len; i++) {
      array[i] = hatch_pattern_to_index(hatch_pattern_prop.get(i))
    }

    this.update(hatch_pattern_prop.is_Scalar())
  }

  set_from_line_cap(line_cap_prop: Uniform<LineCap>): void {
    const len = line_cap_prop.is_Scalar() ? 1 : line_cap_prop.length
    const array = this.get_sized_array(len)

    for (let i = 0; i < len; i++) {
      array[i] = cap_lookup[line_cap_prop.get(i)]
    }

    this.update(line_cap_prop.is_Scalar())
  }

  set_from_line_join(line_join_prop: Uniform<LineJoin>): void {
    const len = line_join_prop.is_Scalar() ? 1 : line_join_prop.length
    const array = this.get_sized_array(len)

    for (let i = 0; i < len; i++) {
      array[i] = join_lookup[line_join_prop.get(i)]
    }

    this.update(line_join_prop.is_Scalar())
  }
}

// Normalized version - values stored as uint8 but read as float 0-1 in shader
export class NormalizedUint8Buffer extends Uint8Buffer {
  // WebGPU handles normalization differently than WebGL
  // We convert to float in the interleaved buffer instead
}
