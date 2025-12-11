""" Scatter plot with 10k points using WebGPU backend.

Note: WebGPU requires browser support. In Chrome, you may need to enable
WebGPU via chrome://flags/#enable-unsafe-webgpu or use Chrome 113+.

"""
import numpy as np

from bokeh.io import output_file
from bokeh.plotting import figure, show

# Use inline resources to ensure we use the local BokehJS build
output_file("scatter10k.html", mode="inline")

N = 10000

x = np.random.normal(0, np.pi, N)
y = np.sin(x) + np.random.normal(0, 0.2, N)

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"

p = figure(tools=TOOLS, output_backend="webgpu")
p.scatter(x, y, alpha=0.1, nonselection_alpha=0.001)

show(p)
