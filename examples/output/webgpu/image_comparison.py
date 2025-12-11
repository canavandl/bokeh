"""Image glyph comparison: Canvas vs WebGL vs WebGPU backends.

Demonstrates the performance and visual comparison of different rendering
backends with multiple images.

Note: WebGPU requires browser support. In Chrome, you may need to enable
WebGPU via chrome://flags/#enable-unsafe-webgpu or use Chrome 113+.
"""
import numpy as np

from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.plotting import figure, show

output_file("image_comparison.html", mode="inline")

# Generate image data - create a grid of images
np.random.seed(42)

# Create a sample image (mandelbrot-like pattern)
def create_sample_image(width, height, offset_x=0, offset_y=0):
    x = np.linspace(-2 + offset_x, 1 + offset_x, width)
    y = np.linspace(-1.5 + offset_y, 1.5 + offset_y, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    c = Z.copy()
    result = np.zeros(Z.shape)

    for i in range(50):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + c[mask]
        result[mask] = i

    return result

# Create multiple images in a grid
N_IMAGES = 16  # 4x4 grid
images = []
x_positions = []
y_positions = []

grid_size = 4
for i in range(grid_size):
    for j in range(grid_size):
        # Create image with slight offset for variety
        img = create_sample_image(200, 200, offset_x=i*0.5 - 1, offset_y=j*0.5 - 1)
        images.append(img)
        x_positions.append(j * 220)
        y_positions.append(i * 220)

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

# Canvas backend
p_canvas = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="canvas",
    title=f"Canvas (n={N_IMAGES})",
    x_range=(-20, 900),
    y_range=(-20, 900),
)
p_canvas.image(
    image=images,
    x=x_positions,
    y=y_positions,
    dw=200,
    dh=200,
    palette="Turbo256",
)

# WebGL backend
p_webgl = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgl",
    title=f"WebGL (n={N_IMAGES})",
    x_range=(-20, 900),
    y_range=(-20, 900),
)
p_webgl.image(
    image=images,
    x=x_positions,
    y=y_positions,
    dw=200,
    dh=200,
    palette="Turbo256",
)

# WebGPU backend
p_webgpu = figure(
    width=400, height=400,
    tools=TOOLS,
    output_backend="webgpu",
    title=f"WebGPU (n={N_IMAGES})",
    x_range=(-20, 900),
    y_range=(-20, 900),
)
p_webgpu.image(
    image=images,
    x=x_positions,
    y=y_positions,
    dw=200,
    dh=200,
    palette="Turbo256",
)

# Show all three side by side
show(row(p_canvas, p_webgl, p_webgpu))
