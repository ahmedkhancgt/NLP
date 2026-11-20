# Chapter 35: Three-Dimensional Plotting in Matplotlib

## Core Idea
The `mplot3d` toolkit extends Matplotlib's 2D machinery to basic 3D visualization (points/lines, contours, wireframes, surfaces, and triangulated surfaces) using nearly the same call signatures as their 2D counterparts, enabled via `projection='3d'`.

## Frameworks Introduced
- **3D axes creation**: `from mpl_toolkits import mplot3d; ax = plt.axes(projection='3d')` — the entry point for all 3D plotting.
- **Points/lines**: `ax.plot3D(x, y, z, ...)`/`ax.scatter3D(x, y, z, c=, cmap=, ...)` — direct 3D analogs of `plt.plot`/`plt.scatter` (Ch 26/27), same keyword conventions.
- **Contours**: `ax.contour3D(X, Y, Z, levels, cmap=)` — requires `X`,`Y`,`Z` as 2D grids (via `np.meshgrid`), same as 2D `plt.contour` (Ch 28).
- **Wireframe/surface**: `ax.plot_wireframe(X, Y, Z)` (line-grid skeleton) vs. `ax.plot_surface(X, Y, Z, rstride=, cstride=, cmap=, edgecolor=)` (filled, colormapped polygons) — both need rectilinear (but not necessarily Cartesian — polar grids work too) 2D input grids.
- **Triangulated surfaces**: `ax.plot_trisurf(x, y, z, cmap=, triangles=)` — for irregularly/randomly sampled (non-gridded) 1D point data; builds a surface by triangulating adjacent points, optionally with an explicit `matplotlib.tri.Triangulation` for correctness on topologically tricky shapes (e.g., a Möbius strip).
- **View control**: `ax.view_init(elev, azim)` sets elevation/azimuth viewing angles explicitly.

## Key Concepts
- **`projection='3d'`**: the keyword that turns any standard axes-creation call into a 3D axes.
- **Interactive viewing benefit**: 3D plots "benefit immensely from viewing figures interactively" (`%matplotlib notebook`) vs. static (`%matplotlib inline`) — depth/rotation is hard to judge in a frozen image.
- **Grid requirement for contour/wireframe/surface**: `X`, `Y`, `Z` must be 2D grids (from `np.meshgrid`), rectilinear but not necessarily rectangular — a polar (r, θ) grid works equally well, mapped to Cartesian x/y before plotting.
- **Triangulation**: the mechanism `plot_trisurf` uses to connect scattered 1D (x,y,z) points into a surface; for topologically subtle shapes, triangulating in the *parameter space* (e.g., θ, w for a Möbius strip) before projecting into 3D avoids visual artifacts.

## Mental Models
- Treat 3D plotting as "the same 2D API plus a Z dimension and a `projection='3d'` axes" — nearly every function/keyword you already know from 2D plotting (Ch 26-28) transfers directly.
- Choose the 3D plot type by data shape: have (x,y,z) triples, possibly scattered? → `scatter3D`/`plot3D`/`plot_trisurf`. Have Z evaluated on a regular (possibly non-Cartesian) grid? → `contour3D`/`plot_wireframe`/`plot_surface`.

## Anti-patterns
- **Assuming `plot_surface`/`plot_wireframe` work on irregular/scattered point data**: they require a proper 2D grid — use `plot_trisurf` for scattered (x,y,z) triples instead.
- **Triangulating directly in 3D embedded space for topologically twisted surfaces (e.g., a Möbius strip)**: can produce visually incorrect/self-intersecting triangulation — triangulate in the underlying 2D parameter space first, then project into 3D.
- **Relying solely on static images to judge 3D plot layout/depth**: much of the intuition of a 3D plot is lost without interactivity — use `%matplotlib notebook` (or equivalent) during exploration.

## Code Examples
```python
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

# points and lines
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline); yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

# contour
def f(x, y): return np.sin(np.sqrt(x**2 + y**2))
x = np.linspace(-6, 6, 30); y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y); Z = f(X, Y)
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 40, cmap='binary')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.view_init(60, 35)                     # elevation, azimuth

# wireframe / surface
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

# polar grid surface
r = np.linspace(0, 6, 20); theta = np.linspace(-0.9*np.pi, 0.8*np.pi, 40)
r, theta = np.meshgrid(r, theta)
X = r * np.sin(theta); Y = r * np.cos(theta); Z = f(X, Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

# triangulated surface from scattered points
theta = 2*np.pi*np.random.random(1000); r = 6*np.random.random(1000)
x = np.ravel(r*np.sin(theta)); y = np.ravel(r*np.cos(theta)); z = f(x, y)
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')

# Mobius strip with parameter-space triangulation
theta = np.linspace(0, 2*np.pi, 30); w = np.linspace(-0.25, 0.25, 8)
w, theta = np.meshgrid(w, theta)
phi = 0.5 * theta
r = 1 + w * np.cos(phi)
x = np.ravel(r * np.cos(theta)); y = np.ravel(r * np.sin(theta)); z = np.ravel(w * np.sin(phi))
from matplotlib.tri import Triangulation
tri = Triangulation(np.ravel(w), np.ravel(theta))
ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap='Greys', linewidths=0.2)
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
```
- **What it demonstrates**: the full progression from basic 3D points/lines through contour/wireframe/surface (gridded) to trisurf (scattered), culminating in correct topological triangulation for a Möbius strip.

## Reference Tables
| Function | Input shape | Output |
|---|---|---|
| `ax.plot3D`/`scatter3D` | 1D (x,y,z) arrays | Lines/points |
| `ax.contour3D` | 2D grids | Contour lines on a 3D relief |
| `ax.plot_wireframe` | 2D grids | Line-grid skeleton surface |
| `ax.plot_surface` | 2D grids (rectilinear, any coordinate system) | Filled colormapped surface |
| `ax.plot_trisurf` | 1D scattered (x,y,z) | Triangulated surface |

## Worked Example
Visualizing a Möbius strip: parametrize by θ (0 to 2π, loop position) and w (-0.25 to 0.25, width), set the twist φ = θ/2 (half-twist per full loop — the defining property of a Möbius strip), compute r = 1 + w·cos(φ) and embed as x=r·cos(θ), y=r·sin(θ), z=w·sin(φ); crucially, triangulate in the (w, θ) parameter space (not the embedded x,y,z space) via `matplotlib.tri.Triangulation`, then pass those `triangles` to `ax.plot_trisurf` — producing a correctly connected single-sided surface that direct 3D-space triangulation would get wrong.

## Key Takeaways
1. 3D plotting requires only `projection='3d'` plus `mpl_toolkits.mplot3d` import — the API otherwise mirrors 2D Matplotlib closely.
2. Points/lines use `plot3D`/`scatter3D`; gridded z=f(x,y) data uses `contour3D`/`plot_wireframe`/`plot_surface`; scattered (non-gridded) data uses `plot_trisurf`.
3. `ax.view_init(elev, azim)` sets the viewing angle explicitly; interactive backends allow click-and-drag rotation.
4. Surface plots accept any rectilinear 2D grid, not just Cartesian — polar grids work directly.
5. For topologically subtle surfaces (e.g., a Möbius strip), triangulate in the underlying 2D parameter space and project the resulting triangulation into 3D, rather than triangulating directly in embedded 3D coordinates.

## Connects To
- **Ch 26/27**: 3D point/line/scatter functions directly mirror their 2D counterparts' call signatures.
- **Ch 28**: 3D contour plots extend the 2D contour chapter's `np.meshgrid`-based approach.
- **Ch 30**: colormap choices and `plt.colorbar()` apply equally to 3D surface plots.
