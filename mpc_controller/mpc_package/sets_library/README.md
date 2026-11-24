# Set library examples

Use these examples to explore the set classes without running the ROS 2
controller. NumPy and SciPy are required; the plotting example also needs
Matplotlib. The project Docker image already includes these dependencies.

From the repository root on the host, open Python with the source package on
the import path:

```bash
PYTHONPATH="$PWD/mpc_controller${PYTHONPATH:+:$PYTHONPATH}" python3 -B
```

Inside the default Compose container, use the mounted source path instead:

```bash
PYTHONPATH="/ros2_ws/src/mpc_controller${PYTHONPATH:+:$PYTHONPATH}" python3 -B
```

Each code block can also be saved as a separate `.py` file and run with the same
`PYTHONPATH` by adding the script name after `python3 -B`. For example, save
the plotting block as `plot_zonotopes.py` and run this from the repository root:

```bash
PYTHONPATH="$PWD/mpc_controller${PYTHONPATH:+:$PYTHONPATH}" python3 -B plot_zonotopes.py
```

## 1. Create a zonotope and check its bounds

A zonotope contains a **center** plus a combination of **generators**. For
`Zonotope`, the columns of `G` are the generators, and their coefficients vary
independently between `-1` and `1`.

```python
import numpy as np
from mpc_package.sets_library import Zonotope

z = Zonotope([0.0, 0.0], [[1.0, 0.5], [0.0, 0.5]])
lower, upper = z.bounds()
print("Lower:", lower.ravel(), "Upper:", upper.ravel())
np.testing.assert_allclose(lower.ravel(), [-1.5, -0.5])
np.testing.assert_allclose(upper.ravel(), [1.5, 0.5])

sample = z.sample(np.random.default_rng(42))
print("Sample:", sample.ravel())
assert z.contains_point(sample)
assert not z.contains_point([3.0, 0.0])
```

`bounds()` returns the smallest axis-aligned box containing the set. A point
inside that box is not necessarily inside the zonotope; use `contains_point()`
to test membership. Prefer `bounds()` for general set calculations.
`envbox_min_max()` is a legacy helper tied to the controller's scheduling limits.

## 2. Map, add, and plot zonotopes

This example maps a 2D zonotope through a matrix, adds a disturbance box, and
plots both sets and the resulting box. It writes a PNG to the current directory
without opening a window, so it also works in headless Docker.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpc_package.sets_library import Zonotope

original = Zonotope([0.0, 0.0], [[1.0, 0.5], [0.0, 0.5]])
disturbance = Zonotope.from_bounds([-0.2, -0.1], [0.2, 0.1])
A = np.array([[0.9, 0.3], [-0.2, 0.8]])

# One prediction step: map the current set, then add the disturbance.
propagated = original.linear_mapping(A) + disturbance
box = propagated.interval_box()

fig, ax = plt.subplots()
for z, label, style in [
    (original, "Original", "-"),
    (propagated, "Mapped + disturbance", "-"),
    (box, "Bounding box", "--"),
]:
    vertices = z.vertices_2d()
    closed = np.vstack((vertices, vertices[0]))
    ax.plot(closed[:, 0], closed[:, 1], style, label=label)

ax.set(xlabel="x1", ylabel="x2", aspect="equal")
ax.grid(True)
ax.legend()
fig.savefig("zonotopes.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

In the default Docker setup, run the plotting script from a writable bind
mount, such as `/ros2_data`, to see the PNG under `data/` on the host.

Other operations to try with the same sets:

| Operation | Example | What it does |
| --- | --- | --- |
| Projection | `propagated.project([0])` | Keeps only the first coordinate. For a higher-dimensional set, use `project([0, 1])` before plotting. |
| Dilation | `original.dilate_box([0.1, 0.2])` | Adds a box with those half-widths. |
| Order reduction | `propagated.reduce_order("box", 2)` | Uses a conservative outer set with fewer generators; useful after repeated additions. |
| Area | `propagated.area_2d()` | Returns the area of the 2D zonotope. |
| Cartesian product | `cartesian_product_zonotope(original, disturbance)` | Stacks two independent sets into a 4D set; import the function from `mpc_package.sets_library`. |

`z1 + z2` is the Minkowski sum. In this library, `z1 - z2` means
`z1 + (-z2)`, **not** constraint tightening. For a box constraint such as
`[-1, 1]` and an error range of `[-0.1, 0.1]`, the tightened bounds are
`[-0.9, 0.9]`. Also, `vertices_2d()` enumerates generator sign combinations,
so keep the generator count small when plotting.

## 3. Try interval and matrix-set operations

An `Interval` stores a lower and upper bound. A `MatrixInterval` gives each
matrix entry its own range. A `MatrixZonotope` uses one coefficient per whole
generator matrix, so entries can vary together.

```python
import numpy as np
from mpc_package.sets_library import MatrixZonotope, Zonotope
from mpc_package.sets_library.intervals import Interval
from mpc_package.sets_library.matrix_interval import MatrixInterval

velocity = Interval(1.0, 3.0)
shifted = velocity + 0.5
np.testing.assert_allclose(shifted.lower, 1.5)
np.testing.assert_allclose(shifted.upper, 3.5)
assert shifted.contains(2.0)

matrix_box = MatrixInterval.from_bounds([[0.9, 0.0], [0.0, 0.8]],
                                       [[1.1, 0.0], [0.0, 1.2]])
print("Matrix interval lower bound:\n", matrix_box.lower)

# One generator matrix with shape (2, 2), stored as a batch of shape (1, 2, 2).
generator = np.diag([0.1, 0.2])
model = MatrixZonotope(np.eye(2), generator[None, :, :])
np.testing.assert_allclose(model.interval_matrix().lower, matrix_box.lower)
np.testing.assert_allclose(model.interval_matrix().upper, matrix_box.upper)

states = Zonotope.from_bounds([-1.0, -0.5], [1.0, 0.5])
reachable = model.zonotope_product(states)  # Outer approximation of M @ x.
lower, upper = reachable.bounds()
print("Reachable box:", lower.ravel(), upper.ravel())
```

The matrix interval and matrix zonotope in this example have the same
entrywise bounds, but they do not describe the same set: the zonotope uses
one shared coefficient for both diagonal entries.
