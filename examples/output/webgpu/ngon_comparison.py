"""Ngon glyph comparison: Canvas vs WebGL vs WebGPU backends.

Demonstrates the performance and visual comparison of different rendering
backends with 10,000 regular polygon (ngon) glyphs with varying numbers of sides.

Note: WebGPU requires browser support. In Chrome, you may need to enable
WebGPU via chrome://flags/#enable-unsafe-webgpu or use Chrome 113+.
"""
import numpy as np

from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.plotting import figure, show

output_file("ngon_comparison.html", mode="inline")

# Generate data - 10k ngons for performance comparison
N = 10000
np.random.seed(42)

# Random positions across the plot
x = np.random.uniform(0, 100, N)
y = np.random.uniform(0, 100, N)

# Random radii
radius = np.random.uniform(0.3, 1.5, N)

# Random number of sides (3 to 8 - triangles, squares, pentagons, hexagons, etc.)
n = np.random.randint(3, 9, N)

# Random rotation angles
angle = np.random.uniform(0, 2 * np.pi, N)

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
p_canvas.ngon(
    x=x, y=y,
    radius=radius,
    n=n,
    angle=angle,
    fill_color="mediumorchid",
    fill_alpha=0.6,
    line_color="indigo",
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
p_webgl.ngon(
    x=x, y=y,
    radius=radius,
    n=n,
    angle=angle,
    fill_color="mediumorchid",
    fill_alpha=0.6,
    line_color="indigo",
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
p_webgpu.ngon(
    x=x, y=y,
    radius=radius,
    n=n,
    angle=angle,
    fill_color="mediumorchid",
    fill_alpha=0.6,
    line_color="indigo",
    line_width=1,
)

# Show all three side by side
show(row(p_canvas, p_webgl, p_webgpu))
