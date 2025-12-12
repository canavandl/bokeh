"""Simple WebGPU glyph test with small number of items for visual inspection."""
import numpy as np
from bokeh.io import output_file, show
from bokeh.layouts import row
from bokeh.plotting import figure

output_file("simple_test.html", mode="inline")

# Small dataset for close inspection
N = 5
np.random.seed(42)
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 3, 5, 4], dtype=float)

TOOLS = "pan,wheel_zoom,box_zoom,reset"

def create_ngon_comparison():
    """Test ngon glyph"""
    plots = []
    for backend in ["canvas", "webgl", "webgpu"]:
        p = figure(width=300, height=300, tools=TOOLS, output_backend=backend,
                   title=f"ngon ({backend})", x_range=(0, 6), y_range=(0, 6))
        p.ngon(x=x, y=y, radius=0.4, n=[3, 4, 5, 6, 7], angle=0,
               fill_color="purple", fill_alpha=0.6, line_color="indigo", line_width=2)
        plots.append(p)
    return row(*plots)

def create_wedge_comparison():
    """Test wedge glyph"""
    plots = []
    for backend in ["canvas", "webgl", "webgpu"]:
        p = figure(width=300, height=300, tools=TOOLS, output_backend=backend,
                   title=f"wedge ({backend})", x_range=(0, 6), y_range=(0, 6))
        p.wedge(x=x, y=y, radius=0.5, start_angle=0.2, end_angle=2.5,
                fill_color="tomato", fill_alpha=0.6, line_color="darkred", line_width=2)
        plots.append(p)
    return row(*plots)

def create_annulus_comparison():
    """Test annulus glyph"""
    plots = []
    for backend in ["canvas", "webgl", "webgpu"]:
        p = figure(width=300, height=300, tools=TOOLS, output_backend=backend,
                   title=f"annulus ({backend})", x_range=(0, 6), y_range=(0, 6))
        p.annulus(x=x, y=y, inner_radius=0.2, outer_radius=0.5,
                  fill_color="steelblue", fill_alpha=0.6, line_color="navy", line_width=2)
        plots.append(p)
    return row(*plots)

def create_annular_wedge_comparison():
    """Test annular_wedge glyph"""
    plots = []
    for backend in ["canvas", "webgl", "webgpu"]:
        p = figure(width=300, height=300, tools=TOOLS, output_backend=backend,
                   title=f"annular_wedge ({backend})", x_range=(0, 6), y_range=(0, 6))
        p.annular_wedge(x=x, y=y, inner_radius=0.2, outer_radius=0.5,
                        start_angle=0.3, end_angle=2.0,
                        fill_color="seagreen", fill_alpha=0.6, line_color="darkgreen", line_width=2)
        plots.append(p)
    return row(*plots)

def create_hex_tile_comparison():
    """Test hex_tile glyph"""
    plots = []
    q = [0, 1, 2, 0, 1]
    r = [0, 0, 0, 1, 1]
    for backend in ["canvas", "webgl", "webgpu"]:
        p = figure(width=300, height=300, tools=TOOLS, output_backend=backend,
                   title=f"hex_tile ({backend})", match_aspect=True)
        p.hex_tile(q=q, r=r, size=1.0, fill_color="steelblue", fill_alpha=0.8,
                   line_color="navy", line_width=2)
        plots.append(p)
    return row(*plots)

if __name__ == "__main__":
    import sys
    glyph = sys.argv[1] if len(sys.argv) > 1 else "ngon"

    if glyph == "ngon":
        show(create_ngon_comparison())
    elif glyph == "wedge":
        show(create_wedge_comparison())
    elif glyph == "annulus":
        show(create_annulus_comparison())
    elif glyph == "annular_wedge":
        show(create_annular_wedge_comparison())
    elif glyph == "hex_tile":
        show(create_hex_tile_comparison())
    else:
        print(f"Unknown glyph: {glyph}")
        print("Available: ngon, wedge, annulus, annular_wedge, hex_tile")
