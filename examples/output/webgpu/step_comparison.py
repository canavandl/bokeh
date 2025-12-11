"""Step glyph comparison: Canvas vs WebGL vs WebGPU backends.

Demonstrates the performance and visual comparison of different rendering
backends with a stepped line containing 50,000 points.

Note: WebGPU requires browser support. In Chrome, you may need to enable
WebGPU via chrome://flags/#enable-unsafe-webgpu or use Chrome 113+.
"""
import numpy as np

from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.plotting import figure, show

output_file("step_comparison.html", mode="inline")

# Generate data - 50k points for performance comparison
N = 50000
np.random.seed(42)

x = np.linspace(0, 100, N)
# Create random walk data (typical for step charts)
y = np.cumsum(np.random.randn(N)) + 50

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

# Canvas backend
p_canvas = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="canvas",
    title=f"Canvas (n={N})",
)
p_canvas.step(x, y, line_width=1, color="navy", mode="after")

# WebGL backend
p_webgl = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgl",
    title=f"WebGL (n={N})",
)
p_webgl.step(x, y, line_width=1, color="navy", mode="after")

# WebGPU backend
p_webgpu = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgpu",
    title=f"WebGPU (n={N})",
)
p_webgpu.step(x, y, line_width=1, color="navy", mode="after")

# Show all three side by side
show(row(p_canvas, p_webgl, p_webgpu))
