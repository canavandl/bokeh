"""Line glyph comparison: Canvas vs WebGL vs WebGPU backends.

Demonstrates the performance and visual comparison of different rendering
backends with a line containing 100,000 points.

Note: WebGPU requires browser support. In Chrome, you may need to enable
WebGPU via chrome://flags/#enable-unsafe-webgpu or use Chrome 113+.
"""
import numpy as np

from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.plotting import figure, show

output_file("line_comparison.html", mode="inline")

# Generate data - 100k points for performance comparison
N = 100000
np.random.seed(42)

x = np.linspace(0, 100, N)
# Create a noisy sine wave
y = np.sin(x * 0.5) * 10 + np.random.normal(0, 0.5, N)

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

# Canvas backend
p_canvas = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="canvas",
    title=f"Canvas (n={N})",
)
p_canvas.line(x, y, line_width=1, color="navy", alpha=0.8)

# WebGL backend
p_webgl = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgl",
    title=f"WebGL (n={N})",
)
p_webgl.line(x, y, line_width=1, color="navy", alpha=0.8)

# WebGPU backend
p_webgpu = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgpu",
    title=f"WebGPU (n={N})",
)
p_webgpu.line(x, y, line_width=1, color="navy", alpha=0.8)

# Show all three side by side
show(row(p_canvas, p_webgl, p_webgpu))
