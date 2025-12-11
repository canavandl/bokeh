"""Hex tile glyph comparison: Canvas vs WebGL vs WebGPU backends.

Demonstrates the performance and visual comparison of different rendering
backends with hexagonal tiles arranged in a grid pattern.

Note: WebGPU requires browser support. In Chrome, you may need to enable
WebGPU via chrome://flags/#enable-unsafe-webgpu or use Chrome 113+.
"""
import numpy as np

from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.plotting import figure, show
from bokeh.util.hex import axial_to_cartesian

output_file("hex_tile_comparison.html", mode="inline")

# Generate hexagonal grid coordinates
# Create a grid of approximately 10k hex tiles
n = 60  # tiles per side
q_coords = []
r_coords = []
for q in range(-n, n + 1):
    for r in range(-n, n + 1):
        # Skip corners to make a roughly circular region
        if abs(q + r) <= n:
            q_coords.append(q)
            r_coords.append(r)

q = np.array(q_coords)
r = np.array(r_coords)
N = len(q)

# Random colors based on position for visual interest
np.random.seed(42)

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

# Convert hex coords to cartesian for appropriate plot ranges
x, y = axial_to_cartesian(q, r, 1.0, "pointytop")
x_range = (x.min() - 1, x.max() + 1)
y_range = (y.min() - 1, y.max() + 1)

# Canvas backend
p_canvas = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="canvas",
    title=f"Canvas (n={N})",
    x_range=x_range,
    y_range=y_range,
    match_aspect=True,
)
p_canvas.hex_tile(
    q=q, r=r,
    size=1.0,
    fill_color="steelblue",
    fill_alpha=0.8,
    line_color="navy",
    line_width=0.5,
)

# WebGL backend
p_webgl = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgl",
    title=f"WebGL (n={N})",
    x_range=x_range,
    y_range=y_range,
    match_aspect=True,
)
p_webgl.hex_tile(
    q=q, r=r,
    size=1.0,
    fill_color="steelblue",
    fill_alpha=0.8,
    line_color="navy",
    line_width=0.5,
)

# WebGPU backend
p_webgpu = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgpu",
    title=f"WebGPU (n={N})",
    x_range=x_range,
    y_range=y_range,
    match_aspect=True,
)
p_webgpu.hex_tile(
    q=q, r=r,
    size=1.0,
    fill_color="steelblue",
    fill_alpha=0.8,
    line_color="navy",
    line_width=0.5,
)

# Show all three side by side
show(row(p_canvas, p_webgl, p_webgpu))
