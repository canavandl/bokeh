"""Rect glyph comparison: Canvas vs WebGL vs WebGPU backends.

Demonstrates the performance and visual comparison of different rendering
backends with 20,000 rectangles.

Note: WebGPU requires browser support. In Chrome, you may need to enable
WebGPU via chrome://flags/#enable-unsafe-webgpu or use Chrome 113+.
"""
import numpy as np

from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.plotting import figure, show

output_file("rect_comparison.html", mode="inline")

# Generate data - 20k rectangles for performance comparison
N = 20000
np.random.seed(42)

x = np.random.uniform(0, 100, N)
y = np.random.uniform(0, 100, N)
width = np.random.uniform(0.5, 2, N)
height = np.random.uniform(0.5, 2, N)
angle = np.random.uniform(0, np.pi, N)
colors = np.random.choice(["red", "green", "blue", "orange", "purple"], N)

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

# Canvas backend
p_canvas = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="canvas",
    title=f"Canvas (n={N})",
)
p_canvas.rect(x, y, width, height, angle=angle, color=colors, alpha=0.5)

# WebGL backend
p_webgl = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgl",
    title=f"WebGL (n={N})",
)
p_webgl.rect(x, y, width, height, angle=angle, color=colors, alpha=0.5)

# WebGPU backend
p_webgpu = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgpu",
    title=f"WebGPU (n={N})",
)
p_webgpu.rect(x, y, width, height, angle=angle, color=colors, alpha=0.5)

# Show all three side by side
show(row(p_canvas, p_webgl, p_webgpu))
