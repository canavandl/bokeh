""" Compare Canvas, WebGL, and WebGPU backends for rendering scatter plots.

This example renders 5000 circles using three different output backends,
allowing you to compare their appearance and interactivity.

Note: WebGPU requires browser support. In Chrome, you may need to enable
WebGPU via chrome://flags/#enable-unsafe-webgpu or use Chrome 113+.

"""
import numpy as np

from bokeh.io import output_file
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, Div
from bokeh.palettes import Turbo256
from bokeh.plotting import figure, show

# Use inline resources to ensure we use the local BokehJS build
output_file("scatter_compare.html", mode="inline")

N = 5000

# Generate random data
rng = np.random.default_rng(42)
x = rng.random(N) * 100
y = rng.random(N) * 100
sizes = 5 + rng.random(N) * 15
colors = [Turbo256[int(i)] for i in rng.integers(0, 256, N)]

source = ColumnDataSource(data=dict(x=x, y=y, sizes=sizes, colors=colors))

def make_plot(title, backend):
    p = figure(
        title=f"{title} (n={N})",
        width=400,
        height=400,
        output_backend=backend,
        x_range=(-5, 105),
        y_range=(-5, 105),
        tools="pan,wheel_zoom,box_zoom,reset",
    )
    p.scatter(
        "x", "y",
        size="sizes",
        fill_color="colors",
        fill_alpha=0.6,
        line_color=None,
        source=source,
    )
    return p

p_canvas = make_plot("Canvas", "canvas")
p_webgl = make_plot("WebGL", "webgl")
p_webgpu = make_plot("WebGPU", "webgpu")

info = Div(text="""
<h3>WebGPU Backend Demo</h3>
<p>This example renders 5000 circles using three different backends:</p>
<ul>
    <li><strong>Canvas:</strong> Standard 2D canvas rendering</li>
    <li><strong>WebGL:</strong> GPU-accelerated via WebGL/ReGL</li>
    <li><strong>WebGPU:</strong> Next-gen GPU API (experimental)</li>
</ul>
<p>Try zooming and panning to compare performance!</p>
<p><em>Note: WebGPU requires browser support. Check your browser console for availability.</em></p>
""", width=400)

show(row(info, p_canvas, p_webgl, p_webgpu))
