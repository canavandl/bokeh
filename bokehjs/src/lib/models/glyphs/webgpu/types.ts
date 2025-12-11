import type {MarkerType} from "core/enums"

// Extended marker types supported by WebGPU backend
export type GPUMarkerType = MarkerType | "rect" | "round_rect" | "ellipse"

// Bounding box for scissor/viewport operations
export type BoundingBox = {
  x: number
  y: number
  width: number
  height: number
}

// Common props passed from GPU glyph classes to render functions
export type CommonProps = {
  scissor: BoundingBox
  viewport: BoundingBox
  canvas_size: [number, number]
}

export type CommonLineProps = CommonProps & {
  antialias: number
}

// Marker glyph props for WebGPU rendering
export type MarkerGlyphProps = CommonLineProps & {
  nmarkers: number
  size_hint: number
  border_radius: [number, number, number, number]
}

// Uniforms structure matching WGSL shader
export type MarkerUniforms = {
  canvas_size: [number, number]
  antialias: number
  size_hint: number
  border_radius: [number, number, number, number]
}

// Transform info passed to draw calls
export type Transform = {
  pixel_ratio: number
  width: number
  height: number
}
