"""MultiLine glyph comparison: Canvas vs WebGL vs WebGPU backends.

Demonstrates the performance and visual comparison of different rendering
backends with 500 lines of 500 points each.

Note: WebGPU requires browser support. In Chrome, you may need to enable
WebGPU via chrome://flags/#enable-unsafe-webgpu or use Chrome 113+.
"""
import numpy as np

from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.plotting import figure, show

output_file("multi_line_comparison.html", mode="inline")

# Generate data - 500 lines with 500 points each
N_LINES = 500
POINTS_PER_LINE = 500
np.random.seed(42)

xs = []
ys = []
colors = []
palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

for i in range(N_LINES):
    x = np.linspace(0, 10, POINTS_PER_LINE) + np.random.uniform(-0.5, 0.5)
    y = np.sin(x + i * 0.1) * 5 + i * 0.02 + np.random.normal(0, 0.1, POINTS_PER_LINE)
    xs.append(x)
    ys.append(y)
    colors.append(palette[i % len(palette)])

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

total_points = N_LINES * POINTS_PER_LINE

# Canvas backend
p_canvas = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="canvas",
    title=f"Canvas ({N_LINES} lines, {total_points} pts)",
)
p_canvas.multi_line(xs, ys, line_color=colors, line_width=1, alpha=0.6)

# WebGL backend
p_webgl = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgl",
    title=f"WebGL ({N_LINES} lines, {total_points} pts)",
)
p_webgl.multi_line(xs, ys, line_color=colors, line_width=1, alpha=0.6)

# WebGPU backend
p_webgpu = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgpu",
    title=f"WebGPU ({N_LINES} lines, {total_points} pts)",
)
p_webgpu.multi_line(xs, ys, line_color=colors, line_width=1, alpha=0.6)

# Show all three side by side
show(row(p_canvas, p_webgl, p_webgpu))
