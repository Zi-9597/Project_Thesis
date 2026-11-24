# Set Library

Reusable uncertainty representations for observer, behavior, and reachability code.

- `zonotope.py`: `Zonotope` construction, arithmetic, projection, containment, sampling, vertices, interval hulls, reduction, and dilation.
- `mat_zonotope.py`: `MatrixZonotope` and uncertain matrix operations.
- `intervals.py`: scalar/vector/array `Interval` arithmetic.
- `matrix_interval.py`: `MatrixInterval` operations and outer approximations for reachability.

Objects are designed to return new set representations rather than mutating shared uncertainty state.
