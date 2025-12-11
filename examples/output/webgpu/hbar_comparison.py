"""Horizontal bar chart comparison: Canvas vs WebGL vs WebGPU backends.

Demonstrates the performance and visual comparison of different rendering
backends with 10,000 horizontal bars.

Note: WebGPU requires browser support. In Chrome, you may need to enable
WebGPU via chrome://flags/#enable-unsafe-webgpu or use Chrome 113+.
"""
import numpy as np

from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.plotting import figure, show

output_file("hbar_comparison.html", mode="inline")

# Generate data - 10k horizontal bars for performance comparison
N = 10000
np.random.seed(42)

y = np.arange(N)
right = np.random.uniform(10, 100, N)
height = 0.8
colors = np.random.choice(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"], N)

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

# Canvas backend
p_canvas = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="canvas",
    title=f"Canvas (n={N})",
    y_range=(0, 200),  # Show subset for visibility
)
p_canvas.hbar(y=y, right=right, height=height, color=colors, alpha=0.7)

# WebGL backend
p_webgl = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgl",
    title=f"WebGL (n={N})",
    y_range=(0, 200),
)
p_webgl.hbar(y=y, right=right, height=height, color=colors, alpha=0.7)

# WebGPU backend
p_webgpu = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgpu",
    title=f"WebGPU (n={N})",
    y_range=(0, 200),
)
p_webgpu.hbar(y=y, right=right, height=height, color=colors, alpha=0.7)

# Show all three side by side
show(row(p_canvas, p_webgl, p_webgpu))
