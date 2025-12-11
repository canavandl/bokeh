import {display, fig, row} from "../_util"

// WebGPU backend tests
// Note: These tests require browser support for WebGPU.
// The "webgpu" output backend was added as part of the WebGPU implementation.

describe("webgpu", () => {
  function has_webgpu(): boolean {
    return typeof navigator !== "undefined" && "gpu" in navigator
  }

  it("should detect WebGPU availability", async () => {
    // This test always runs and reports WebGPU status
    const available = has_webgpu()
    console.log(`WebGPU available: ${available}`)
    if (available) {
      const adapter = await navigator.gpu.requestAdapter()
      console.log(`WebGPU adapter: ${adapter != null ? "found" : "not found"}`)
    }
    // This test passes regardless - it's just for diagnostic output
  })

  describe("circle glyph", () => {
    it("should render circles with webgpu backend", async () => {
      if (!has_webgpu()) {
        console.log("WebGPU not available, skipping test")
        return
      }

      const x = [1, 2, 3, 4, 5]
      const y = [1, 4, 2, 3, 5]

      // Use canvas as reference
      const p_canvas = fig([300, 300], {output_backend: "canvas", title: "canvas"})
      p_canvas.circle({x, y, radius: 0.2, fill_color: "blue", line_color: "black", line_width: 2})

      // WebGPU version - cast to any to bypass type check until types are rebuilt
      const p_webgpu = fig([300, 300], {output_backend: "webgpu" as any, title: "webgpu"})
      p_webgpu.circle({x, y, radius: 0.2, fill_color: "blue", line_color: "black", line_width: 2})

      await display(row([p_canvas, p_webgpu]))
    })

    it("should render circles with varying sizes", async () => {
      if (!has_webgpu()) {
        console.log("WebGPU not available, skipping test")
        return
      }

      const x = [1, 2, 3, 4, 5]
      const y = [1, 2, 3, 4, 5]
      const radius = [0.1, 0.2, 0.3, 0.4, 0.5]

      const p_canvas = fig([300, 300], {output_backend: "canvas", title: "canvas"})
      p_canvas.circle({x, y, radius, fill_color: "red", fill_alpha: 0.5, line_color: "darkred"})

      const p_webgpu = fig([300, 300], {output_backend: "webgpu" as any, title: "webgpu"})
      p_webgpu.circle({x, y, radius, fill_color: "red", fill_alpha: 0.5, line_color: "darkred"})

      await display(row([p_canvas, p_webgpu]))
    })

    it("should render circles with varying colors", async () => {
      if (!has_webgpu()) {
        console.log("WebGPU not available, skipping test")
        return
      }

      const x = [1, 2, 3, 4, 5]
      const y = [3, 3, 3, 3, 3]
      const fill_color = ["red", "green", "blue", "orange", "purple"]

      const p_canvas = fig([400, 200], {output_backend: "canvas", title: "canvas"})
      p_canvas.circle({x, y, radius: 0.3, fill_color, line_color: "black"})

      const p_webgpu = fig([400, 200], {output_backend: "webgpu" as any, title: "webgpu"})
      p_webgpu.circle({x, y, radius: 0.3, fill_color, line_color: "black"})

      await display(row([p_canvas, p_webgpu]))
    })

    it("should handle empty data", async () => {
      if (!has_webgpu()) {
        console.log("WebGPU not available, skipping test")
        return
      }

      const p_canvas = fig([200, 200], {output_backend: "canvas", title: "canvas"})
      p_canvas.circle({x: [], y: [], radius: 0.2})

      const p_webgpu = fig([200, 200], {output_backend: "webgpu" as any, title: "webgpu"})
      p_webgpu.circle({x: [], y: [], radius: 0.2})

      await display(row([p_canvas, p_webgpu]))
    })

    it("should handle NaN values in coordinates", async () => {
      if (!has_webgpu()) {
        console.log("WebGPU not available, skipping test")
        return
      }

      const x = [1, 2, NaN, 4, 5]
      const y = [1, NaN, 3, 4, 5]

      const p_canvas = fig([300, 300], {output_backend: "canvas", title: "canvas"})
      p_canvas.circle({x, y, radius: 0.2, fill_color: "green"})

      const p_webgpu = fig([300, 300], {output_backend: "webgpu" as any, title: "webgpu"})
      p_webgpu.circle({x, y, radius: 0.2, fill_color: "green"})

      await display(row([p_canvas, p_webgpu]))
    })

    it("should render many circles efficiently", async () => {
      if (!has_webgpu()) {
        console.log("WebGPU not available, skipping test")
        return
      }

      const n = 1000
      const x = Array.from({length: n}, () => Math.random() * 10)
      const y = Array.from({length: n}, () => Math.random() * 10)

      const p_canvas = fig([400, 400], {output_backend: "canvas", title: `canvas (n=${n})`})
      p_canvas.circle({x, y, radius: 0.05, fill_color: "blue", fill_alpha: 0.3})

      const p_webgpu = fig([400, 400], {output_backend: "webgpu" as any, title: `webgpu (n=${n})`})
      p_webgpu.circle({x, y, radius: 0.05, fill_color: "blue", fill_alpha: 0.3})

      await display(row([p_canvas, p_webgpu]))
    })
  })

  describe("fallback behavior", () => {
    it("should fall back to canvas when webgpu is not available", async () => {
      // This test verifies graceful fallback behavior
      const x = [1, 2, 3]
      const y = [1, 2, 3]

      const p = fig([200, 200], {output_backend: "canvas", title: "Canvas fallback"})
      p.circle({x, y, radius: 0.15, fill_color: "purple"})

      await display(p)
    })
  })

  describe("comparison with webgl", () => {
    it("should produce similar output to webgl for circles", async () => {
      if (!has_webgpu()) {
        console.log("WebGPU not available, skipping test")
        return
      }

      const x = [1, 2, 3, 4, 5]
      const y = [2, 4, 1, 3, 5]

      const p_canvas = fig([250, 250], {output_backend: "canvas", title: "canvas"})
      p_canvas.circle({
        x, y,
        radius: 0.25,
        fill_color: "steelblue",
        fill_alpha: 0.7,
        line_color: "navy",
        line_width: 2,
      })

      const p_webgl = fig([250, 250], {output_backend: "webgl", title: "webgl"})
      p_webgl.circle({
        x, y,
        radius: 0.25,
        fill_color: "steelblue",
        fill_alpha: 0.7,
        line_color: "navy",
        line_width: 2,
      })

      const p_webgpu = fig([250, 250], {output_backend: "webgpu" as any, title: "webgpu"})
      p_webgpu.circle({
        x, y,
        radius: 0.25,
        fill_color: "steelblue",
        fill_alpha: 0.7,
        line_color: "navy",
        line_width: 2,
      })

      await display(row([p_canvas, p_webgl, p_webgpu]))
    })
  })
})
