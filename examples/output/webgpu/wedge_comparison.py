"""Wedge glyph comparison: Canvas vs WebGL vs WebGPU backends.

Demonstrates the performance and visual comparison of different rendering
backends with 10,000 pie-slice shaped wedges.

Note: WebGPU requires browser support. In Chrome, you may need to enable
WebGPU via chrome://flags/#enable-unsafe-webgpu or use Chrome 113+.
"""
import numpy as np

from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.plotting import figure, show

output_file("wedge_comparison.html", mode="inline")

# Generate data - 10k wedges for performance comparison
N = 10000
np.random.seed(42)

# Random positions across the plot
x = np.random.uniform(0, 100, N)
y = np.random.uniform(0, 100, N)

# Random radii
radius = np.random.uniform(0.5, 2.0, N)

# Random start and end angles (creating varied pie slices)
start_angle = np.random.uniform(0, 2 * np.pi, N)
# End angle is start + some arc (between pi/6 and pi)
end_angle = start_angle + np.random.uniform(np.pi / 6, np.pi, N)

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
p_canvas.wedge(
    x=x, y=y,
    radius=radius,
    start_angle=start_angle,
    end_angle=end_angle,
    fill_color="tomato",
    fill_alpha=0.6,
    line_color="darkred",
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
p_webgl.wedge(
    x=x, y=y,
    radius=radius,
    start_angle=start_angle,
    end_angle=end_angle,
    fill_color="tomato",
    fill_alpha=0.6,
    line_color="darkred",
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
p_webgpu.wedge(
    x=x, y=y,
    radius=radius,
    start_angle=start_angle,
    end_angle=end_angle,
    fill_color="tomato",
    fill_alpha=0.6,
    line_color="darkred",
    line_width=1,
)

# Show all three side by side
show(row(p_canvas, p_webgl, p_webgpu))
