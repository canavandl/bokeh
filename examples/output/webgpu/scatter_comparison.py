"""Scatter plot comparison: Canvas vs WebGL vs WebGPU backends.

Demonstrates the performance and visual comparison of different rendering
backends with 50,000 scatter points.

Note: WebGPU requires browser support. In Chrome, you may need to enable
WebGPU via chrome://flags/#enable-unsafe-webgpu or use Chrome 113+.
"""
import numpy as np

from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.plotting import figure, show

output_file("scatter_comparison.html", mode="inline")

# Generate data - 50k points for performance comparison
N = 50000
np.random.seed(42)

x = np.random.normal(0, 1, N)
y = np.random.normal(0, 1, N)
colors = np.random.choice(["red", "green", "blue", "orange", "purple"], N)
sizes = np.random.uniform(3, 10, N)

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"

# Canvas backend (CPU rendering)
p_canvas = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="canvas",
    title=f"Canvas (n={N})",
)
p_canvas.scatter(x, y, color=colors, size=sizes, alpha=0.5)

# WebGL backend (GPU rendering via WebGL)
p_webgl = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgl",
    title=f"WebGL (n={N})",
)
p_webgl.scatter(x, y, color=colors, size=sizes, alpha=0.5)

# WebGPU backend (GPU rendering via WebGPU)
p_webgpu = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgpu",
    title=f"WebGPU (n={N})",
)
p_webgpu.scatter(x, y, color=colors, size=sizes, alpha=0.5)

# Show all three side by side
show(row(p_canvas, p_webgl, p_webgpu))
