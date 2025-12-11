"""Annulus glyph comparison: Canvas vs WebGL vs WebGPU backends.

Demonstrates the performance and visual comparison of different rendering
backends with 10,000 ring/donut shaped annuli.

Note: WebGPU requires browser support. In Chrome, you may need to enable
WebGPU via chrome://flags/#enable-unsafe-webgpu or use Chrome 113+.
"""
import numpy as np

from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.plotting import figure, show

output_file("annulus_comparison.html", mode="inline")

# Generate data - 10k annuli for performance comparison
N = 10000
np.random.seed(42)

# Random positions across the plot
x = np.random.uniform(0, 100, N)
y = np.random.uniform(0, 100, N)

# Random inner and outer radii (outer > inner)
inner_radius = np.random.uniform(0.3, 1.0, N)
outer_radius = inner_radius + np.random.uniform(0.2, 0.8, N)

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

# Canvas backend
p_canvas = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="canvas",
    title=f"Canvas (n={N})",
    x_range=(0, 100),
    y_range=(0, 100),
)
p_canvas.annulus(
    x=x, y=y,
    inner_radius=inner_radius,
    outer_radius=outer_radius,
    fill_color="steelblue",
    fill_alpha=0.6,
    line_color="navy",
    line_width=1,
)

# WebGL backend
p_webgl = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgl",
    title=f"WebGL (n={N})",
    x_range=(0, 100),
    y_range=(0, 100),
)
p_webgl.annulus(
    x=x, y=y,
    inner_radius=inner_radius,
    outer_radius=outer_radius,
    fill_color="steelblue",
    fill_alpha=0.6,
    line_color="navy",
    line_width=1,
)

# WebGPU backend
p_webgpu = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgpu",
    title=f"WebGPU (n={N})",
    x_range=(0, 100),
    y_range=(0, 100),
)
p_webgpu.annulus(
    x=x, y=y,
    inner_radius=inner_radius,
    outer_radius=outer_radius,
    fill_color="steelblue",
    fill_alpha=0.6,
    line_color="navy",
    line_width=1,
)

# Show all three side by side
show(row(p_canvas, p_webgl, p_webgpu))
