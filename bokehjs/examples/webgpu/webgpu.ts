import Bokeh from "/static/js/bokeh.esm.js"
import "/static/js/bokeh-gl.esm.js"
import "/static/js/bokeh-api.esm.js"

export namespace WebGPUExample {
  console.log(`Bokeh ${Bokeh.version}`)
  Bokeh.set_log_level("info")

  // Check WebGPU availability
  const has_webgpu = "gpu" in navigator
  console.log(`WebGPU available: ${has_webgpu}`)

  // Generate random data for scatter plot
  const n = 5000
  const x = Array.from({length: n}, () => Math.random() * 100)
  const y = Array.from({length: n}, () => Math.random() * 100)
  const sizes = Array.from({length: n}, () => 5 + Math.random() * 15)
  const colors = Array.from({length: n}, () => {
    const r = Math.floor(Math.random() * 256)
    const g = Math.floor(Math.random() * 256)
    const b = Math.floor(Math.random() * 256)
    return `rgb(${r},${g},${b})`
  })

  const source = new Bokeh.ColumnDataSource({
    data: {x, y, sizes, colors},
  })

  // Create three plots with different backends for comparison
  function make_plot(title: string, output_backend: string): Bokeh.Figure {
    const p = Bokeh.Plotting.figure({
      title: `${title} (n=${n})`,
      width: 400,
      height: 400,
      output_backend: output_backend as any,
      x_range: [-5, 105],
      y_range: [-5, 105],
    })

    p.scatter({
      x: {field: "x"},
      y: {field: "y"},
      size: {field: "sizes"},
      fill_color: {field: "colors"},
      fill_alpha: 0.6,
      line_color: null,
      source,
    })

    return p
  }

  const p_canvas = make_plot("Canvas", "canvas")
  const p_webgl = make_plot("WebGL", "webgl")
  const p_webgpu = make_plot("WebGPU", "webgpu")

  // Layout - use gridplot directly without Div widget
  const grid = Bokeh.Plotting.gridplot(
    [[p_canvas, p_webgl, p_webgpu]],
    {toolbar_location: "above"},
  )

  const doc = new Bokeh.Document()
  doc.add_root(grid)

  const div = document.getElementById("plot")!
  void Bokeh.embed.add_document_standalone(doc, div)
}
