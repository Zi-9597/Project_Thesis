"""General-purpose ordinary zonotope operations.

Mathematical representation
---------------------------
An ordinary zonotope is represented as

    Z = <c, G> = { c + G @ xi : xi in [-1, 1]^p }.

The center has shape ``(n, 1)`` and the generator matrix has shape ``(n, p)``.
This module contains mathematics only. It deliberately has no Matplotlib,
observer, trajectory, command-line, or occupancy-propagation code.

Paper-oriented operations
--------------------------
The later behavior-aware occupancy algorithm uses this generic class to:

* append control-margin generators in Eq. (15);
* obtain [theta], [velocity], and [curvature] for Eqs. (18)-(19);
* construct future state sets after Eqs. (16)-(20);
* project the future state set onto [p_x, p_y] after Eq. (20);
* remove, merge, reduce, and dilate occupancy generators.
"""

from __future__ import annotations

from itertools import combinations, product
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

ReductionMethod = Literal["box", "pca", "pca-box"]


class Zonotope:
    """Represent ``Z = <c, G> = {c + G @ xi : xi in [-1, 1]^p}``.

    The class follows the project convention ``center.shape == (n, 1)`` and
    ``generators.shape == (n, p)``. Every operation returns a new zonotope.

    Inputs
    ------
    ``center`` is the nominal point of the set. It is reshaped to a column
    vector with shape ``(n, 1)``.

    ``Gen`` is the generator matrix. Each column is one independent segment
    direction and must have shape ``(n, p)``.

    Output
    ------
    A ``Zonotope`` object storing a copy of the normalized center and generator
    matrix.

    Example
    -------
    ``Zonotope([[0.0], [1.0]], [[1.0, 0.0], [0.0, 2.0]])`` represents
    all points ``[0, 1]^T + G @ xi`` with ``xi in [-1, 1]^2``.
    """

    def __init__(self, center: np.ndarray, Gen: np.ndarray) -> None:
        """Construct and validate ``Z = <center, Gen>``.

        The constructor is used for state, input, noise, and occupancy sets.
        An empty matrix with shape ``(n, 0)`` is valid and represents a
        singleton ``{center}``.

        Inputs
        ------
        ``center``: scalar-like, vector-like, or matrix-like numeric input that
        can be reshaped to ``(n, 1)``.

        ``Gen``: finite two-dimensional numeric matrix with ``n`` rows.

        Output
        ------
        Initializes ``self._center`` and ``self._generators``.

        Example
        -------
        ``Zonotope([[1.0], [2.0]], np.eye(2))`` creates a two-dimensional box
        centered at ``[1, 2]^T`` with unit coordinate generators.
        """
        center_array = np.asarray(center, dtype=float).reshape(-1, 1)
        generator_array = np.asarray(Gen, dtype=float)

        if generator_array.ndim != 2:
            raise ValueError("Gen must be a two-dimensional matrix with shape (n, p).")
        if center_array.shape[0] != generator_array.shape[0]:
            raise ValueError(
                "Center and generator matrix must have the same number of rows. "
                f"Received {center_array.shape} and {generator_array.shape}."
            )
        if not np.all(np.isfinite(center_array)):
            raise ValueError("Center contains NaN or infinity.")
        if not np.all(np.isfinite(generator_array)):
            raise ValueError("Generator matrix contains NaN or infinity.")

        self._center = center_array.copy()
        self._generators = generator_array.copy()

    @property
    def center(self) -> np.ndarray:
        """Return the center ``c`` with shape ``(n, 1)``."""
        return self._center

    @property
    def generators(self) -> np.ndarray:
        """Return the generator matrix ``G`` with shape ``(n, p)``."""
        return self._generators

    @property
    def c(self) -> np.ndarray:
        """Compatibility alias for ``center``."""
        return self.center

    @property
    def G(self) -> np.ndarray:
        """Compatibility alias for ``generators``."""
        return self.generators

    @property
    def Z(self) -> np.ndarray:
        """Return the computed combined matrix ``[c | G]``."""
        return np.hstack((self.center, self.generators))

    @property
    def n(self) -> int:
        """Return the ambient dimension ``n``."""
        return self.center.shape[0]

    @property
    def p(self) -> int:
        """Return the number of generators ``p``."""
        return self.generators.shape[1]

    @property
    def num_generators(self) -> int:
        """Readable alias for ``p``."""
        return self.p

    def __repr__(self) -> str:
        """Return a useful developer representation."""
        return f"Zonotope(center={self.center!r}, Gen={self.generators!r})"

    def copy(self) -> "Zonotope":
        """Return a separate object representing the same set."""
        return Zonotope(self.center.copy(), self.generators.copy())

    def order(self) -> float:
        """Return the order ``rho(Z) = p / n`` without integer truncation.

        Paper relationship
        -------------------
        The paper does not assign an equation number to order. It is useful for
        monitoring generator growth during repeated reachability propagation.
        """
        if self.n == 0:
            raise ZeroDivisionError("A zero-dimensional zonotope has no order.")
        return self.p / self.n

    def __add__(self, other: "Zonotope") -> "Zonotope":
        """Return the Minkowski sum.

        Equation
        --------
        For ``Z1=<c1,G1>`` and ``Z2=<c2,G2>``:

            Z1 (+) Z2 = <c1+c2, [G1 | G2]>.

        Paper relationship
        -------------------
        The operation is used by dilation and later set propagation. It
        preserves independent uncertainty coefficients by concatenating G1/G2.

        Inputs
        ------
        ``self`` and ``other`` must be zonotopes with the same ambient
        dimension ``n``.

        Output
        ------
        A new zonotope with center ``self.center + other.center`` and generator
        matrix ``[self.generators | other.generators]``.

        Example
        -------
        ``Zonotope([[0.0]], [[1.0]]) + Zonotope([[2.0]], [[0.5]])`` returns
        ``<[[2.0]], [[1.0, 0.5]]>``.
        """
        if not isinstance(other, Zonotope):
            return NotImplemented
        self._require_same_dimension(other, "Minkowski addition")
        return Zonotope(self.center + other.center, np.hstack((self.generators, other.generators)))

    def __sub__(self, other: "Zonotope") -> "Zonotope":
        """Subtract another zonotope using the preserved project convention.

        Equation
        --------
        ``Z1 - Z2 = <c1-c2, [G1 | -G2]>``.

        This equals ``Z1 (+) (-Z2)``. The core occupancy chain does not need
        subtraction, but the operator is preserved for project compatibility.

        Inputs
        ------
        ``self`` and ``other`` must be zonotopes with the same dimension.

        Output
        ------
        A new zonotope with center ``self.center - other.center`` and generator
        matrix ``[self.generators | -other.generators]``.

        Example
        -------
        ``Zonotope([[2.0]], [[1.0]]) - Zonotope([[0.5]], [[0.2]])`` returns a
        zonotope centered at ``1.5`` with generators ``[1.0, -0.2]``.
        """
        if not isinstance(other, Zonotope):
            return NotImplemented
        self._require_same_dimension(other, "Zonotope subtraction")
        return Zonotope(self.center - other.center, np.hstack((self.generators, -other.generators)))

    @classmethod
    def from_bounds(cls, lower: np.ndarray, upper: np.ndarray) -> "Zonotope":
        """Construct an axis-aligned box from coordinate bounds.

        Equations
        ---------
        ``c = (lower + upper) / 2``

        ``G = diag((upper - lower) / 2)``

        Paper relationship
        -------------------
        This is a generic constructor used later for state-error, noise, and
        fixed input-set baselines; it is not a numbered paper equation.

        Inputs
        ------
        ``lower`` and ``upper`` are coordinate bounds with the same shape.

        Output
        ------
        A box zonotope whose coordinate interval hull is exactly
        ``[lower, upper]``.

        Example
        -------
        ``Zonotope.from_bounds([-1.0, 2.0], [3.0, 6.0])`` gives center
        ``[[1.0], [4.0]]`` and generators ``diag([2.0, 2.0])``.
        """
        lower_array = np.asarray(lower, dtype=float).reshape(-1, 1)
        upper_array = np.asarray(upper, dtype=float).reshape(-1, 1)
        if lower_array.shape != upper_array.shape:
            raise ValueError("lower and upper must have identical shapes.")
        if not np.all(np.isfinite(lower_array)) or not np.all(np.isfinite(upper_array)):
            raise ValueError("Bounds must contain finite values.")
        if np.any(lower_array > upper_array):
            raise ValueError("Every lower bound must be less than or equal to its upper bound.")
        center = 0.5 * (lower_array + upper_array)
        radius = 0.5 * (upper_array - lower_array)
        return cls(center, np.diag(radius.ravel()))

    @classmethod
    def from_control_observations(
        cls,
        observations: np.ndarray,
        generators: np.ndarray,
        margin: np.ndarray | None = None,
    ) -> "Zonotope":
        """Fit the paper's fixed-direction control zonotope with an LP.

        For observations ``u_j`` and unit-direction columns ``g_i``, this solves

            minimize sum(alpha_i)
            subject to u_j = c + G_delta @ delta_j
            and -alpha_i <= delta_ij <= alpha_i.

        The returned generator block is ``g_i * alpha_i``. ``margin`` contains
        optional extra generator columns, as required by Eq. (15).
        Observations are rows; controls must have the same dimension as the
        generator rows.
        """
        points = np.asarray(observations, dtype=float)
        directions = np.asarray(generators, dtype=float)
        if points.ndim != 2 or directions.ndim != 2:
            raise ValueError("observations and generators must be two-dimensional.")
        if points.shape[1] != directions.shape[0]:
            raise ValueError("Observation and generator dimensions do not match.")
        if points.shape[0] == 0 or directions.shape[1] == 0:
            raise ValueError("At least one observation and generator are required.")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(directions)):
            raise ValueError("observations and generators must be finite.")

        n_samples, dimension = points.shape
        n_generators = directions.shape[1]
        # Variables are [center(dimension), alpha(n_generators), delta(n_generators*n_samples)].
        center_slice = slice(0, dimension)
        alpha_slice = slice(dimension, dimension + n_generators)
        delta_start = dimension + n_generators
        delta_index = lambda i, j: delta_start + i * n_samples + j
        n_variables = delta_start + n_generators * n_samples

        objective = np.zeros(n_variables)
        objective[alpha_slice] = 1.0
        equality = np.zeros((dimension * n_samples, n_variables))
        rhs = np.zeros(dimension * n_samples)
        row = 0
        for j in range(n_samples):
            for coordinate in range(dimension):
                equality[row, coordinate] = 1.0
                for i in range(n_generators):
                    equality[row, delta_index(i, j)] = -directions[coordinate, i]
                rhs[row] = points[j, coordinate]
                row += 1

        bounds = [(None, None)] * dimension
        bounds.extend([(0.0, None)] * n_generators)
        bounds.extend([(-1.0, 1.0)] * (n_generators * n_samples))
        result = linprog(objective, A_eq=equality, b_eq=rhs, bounds=bounds, method="highs")
        if not result.success:
            raise ValueError(f"Could not fit control zonotope: {result.message}")

        center = result.x[center_slice].reshape(-1, 1)
        scaled_generators = directions * result.x[alpha_slice].reshape(1, -1)
        fitted = cls(center, scaled_generators)
        if margin is not None:
            return fitted.append_generators(np.asarray(margin, dtype=float))
        return fitted

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact coordinate-wise interval hull.

        Equations
        ---------
        ``r = |G| 1``

        ``lower = c - r``

        ``upper = c + r``

        Safety rule
        -----------
        The radius is ``sum(abs(G), axis=1)``. Never use
        ``abs(sum(G, axis=1))`` because opposite-signed generators still enlarge
        the represented set.

        Paper relationship
        -------------------
        Supplies ``[theta]``, ``[velocity]``, and ``[curvature]`` for paper
        Eqs. (18)-(19).

        Inputs
        ------
        Uses the current zonotope ``<c, G>``.

        Output
        ------
        ``(lower, upper)`` where both arrays have shape ``(n, 1)``.

        Example
        -------
        For ``Zonotope([[0.0]], [[1.0, -2.0]])``, the radius is ``3.0`` and
        the returned bounds are ``lower=[[-3.0]]`` and ``upper=[[3.0]]``.
        """
        radius = np.sum(np.abs(self.generators), axis=1, keepdims=True)
        return self.center - radius, self.center + radius

    def interval_box(self) -> "Zonotope":
        """Return ``box(Z) = <c, diag(|G|1)>``.

        This is a conservative outer approximation. The paper later uses the
        same box idea when discarded occupancy generators are reduced.

        Inputs
        ------
        Uses the current zonotope ``<c, G>``.

        Output
        ------
        A new axis-aligned box zonotope with the same center and coordinate
        radii ``sum(abs(G), axis=1)``.

        Example
        -------
        If ``G = [[1.0, -2.0], [0.5, 0.5]]``, then the box generators are
        ``diag([3.0, 1.0])``.
        """
        radius = np.sum(np.abs(self.generators), axis=1)
        return Zonotope(self.center, np.diag(radius))

    def linear_mapping(self, matrix: np.ndarray) -> "Zonotope":
        """Apply ``A Z = <A c, A G>`` exactly.

        Paper relationship
        -------------------
        This generic operation supports future state-set transformations and is
        also the internal operation used by ``project`` after Eq. (20).

        Inputs
        ------
        ``matrix`` is a finite numeric matrix with shape ``(m, n)``.

        Output
        ------
        A new zonotope in dimension ``m`` with center ``matrix @ center`` and
        generators ``matrix @ generators``.

        Example
        -------
        With ``matrix=[[1.0, 0.0]]``, a two-dimensional zonotope is mapped to
        its first coordinate.
        """
        matrix_array = np.asarray(matrix, dtype=float)
        if matrix_array.ndim != 2:
            raise ValueError("The mapping matrix must be two-dimensional.")
        if matrix_array.shape[1] != self.n:
            raise ValueError("The mapping matrix has an incompatible input dimension.")
        if not np.all(np.isfinite(matrix_array)):
            raise ValueError("The mapping matrix contains NaN or infinity.")
        return Zonotope(matrix_array @ self.center, matrix_array @ self.generators)

    def append_generators(self, generators: np.ndarray) -> "Zonotope":
        """Return ``<c, [G | G_new]>`` with independent new generators.

        Paper Eq. (15)
        ---------------
        The behavior-aware input set has the structure

            H_u = [alpha_1 g_1, ..., alpha_ng g_ng, g_u1, g_u2].

        The LP provides the first block. This method appends the two behavioral
        margin columns without changing the center.

        Inputs
        ------
        ``generators`` is a finite matrix with shape ``(n, q)``.

        Output
        ------
        A new zonotope with the same center and ``p + q`` generators.

        Example
        -------
        ``Zonotope([[0.0]], [[1.0]]).append_generators([[0.2, 0.3]])`` returns
        a zonotope with generator matrix ``[[1.0, 0.2, 0.3]]``.
        """
        generator_array = np.asarray(generators, dtype=float)
        if generator_array.ndim != 2:
            raise ValueError("New generators must be a two-dimensional matrix.")
        if generator_array.shape[0] != self.n:
            raise ValueError("New generators have the wrong ambient dimension.")
        if not np.all(np.isfinite(generator_array)):
            raise ValueError("New generators contain NaN or infinity.")
        return Zonotope(self.center, np.hstack((self.generators, generator_array)))

    def project(self, indices: Iterable[int]) -> "Zonotope":
        """Project onto selected coordinates using ``P Z = <P c, P G>``.

        Paper Eq. (20)
        ---------------
        For state order ``[p_x, p_y, theta, v]``, occupancy extraction uses

            P_xy = [[1,0,0,0], [0,1,0,0]],

        implemented as ``future_state_set.project([0, 1])``.

        Inputs
        ------
        ``indices`` is an iterable of unique coordinate indices to keep.

        Output
        ------
        A new lower-dimensional zonotope containing only the selected rows of
        the center and generator matrix.

        Example
        -------
        For state order ``[p_x, p_y, theta, v]``, ``project([0, 1])`` returns
        the position zonotope ``[p_x, p_y]``.
        """
        selected = list(indices)
        if not selected:
            raise ValueError("At least one projection index is required.")
        if any(not isinstance(index, (int, np.integer)) for index in selected):
            raise TypeError("Projection indices must be integers.")
        if len(set(selected)) != len(selected):
            raise ValueError("Projection indices must be unique.")
        if min(selected) < 0 or max(selected) >= self.n:
            raise IndexError("Projection index is outside the set dimension.")
        return self.linear_mapping(np.eye(self.n)[selected, :])

    def remove_zero_generators(self, tolerance: float = 1e-10) -> "Zonotope":
        """Remove columns satisfying ``||g_j||_2 <= tolerance``.

        Paper relationship
        -------------------
        This is the first occupancy simplification after projecting the
        four-dimensional future state set to position coordinates. Heading-only
        or velocity-only generators become zero after projection.

        Inputs
        ------
        ``tolerance`` is a non-negative norm threshold.

        Output
        ------
        A new zonotope with zero or near-zero generator columns removed.

        Example
        -------
        If ``G = [[1.0, 0.0], [0.0, 0.0]]`` and ``tolerance=1e-10``, the
        second generator is removed.
        """
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        keep = np.linalg.norm(self.generators, axis=0) > tolerance
        return Zonotope(self.center, self.generators[:, keep])

    def merge_parallel_generators(self, tolerance: float = 1e-10) -> "Zonotope":
        """Merge parallel and antiparallel generators without under-approximation.

        Mathematical rule
        ------------------
        Because ``g[-1,1] = (-g)[-1,1]``, collinear generators can be replaced
        by one generator in the same direction whose length is the sum of the
        absolute projected lengths.

        Paper relationship
        -------------------
        This is the second occupancy simplification after zero-generator
        removal.

        Inputs
        ------
        ``tolerance`` controls when two generator directions are considered
        collinear.

        Output
        ------
        A new zonotope with collinear generators merged into one longer
        generator per direction.

        Example
        -------
        In 2D, generators ``[1, 0]^T`` and ``[-2, 0]^T`` are merged into one
        generator along the x-axis with length ``3``.
        """
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        cleaned = self.remove_zero_generators(tolerance)
        used = np.zeros(cleaned.p, dtype=bool)
        merged: list[np.ndarray] = []

        for first_index in range(cleaned.p):
            if used[first_index]:
                continue
            reference = cleaned.generators[:, first_index]
            direction = reference / np.linalg.norm(reference)
            total_length = np.linalg.norm(reference)
            used[first_index] = True

            for candidate_index in range(first_index + 1, cleaned.p):
                if used[candidate_index]:
                    continue
                candidate = cleaned.generators[:, candidate_index]
                projection = float(direction @ candidate)
                residual = candidate - projection * direction
                scale = max(1.0, np.linalg.norm(candidate))
                if np.linalg.norm(residual) <= tolerance * scale:
                    total_length += abs(projection)
                    used[candidate_index] = True

            merged.append(direction * total_length)

        matrix = np.column_stack(merged) if merged else np.empty((self.n, 0))
        return Zonotope(cleaned.center, matrix)

    def reduce_order_outer_box(self, max_generators: int) -> "Zonotope":
        """Reduce generator count with a guaranteed axis-aligned outer box.

        Algorithm and equation
        -----------------------
        Keep the largest ``max_generators - n`` columns. For discarded ``G_d``:

            r_d = |G_d| 1,
            G_box = diag(r_d).

        Return ``<c, [G_kept | G_box]>``.

        Paper relationship
        -------------------
        This is the third occupancy simplification. Use this method for direct
        paper-oriented reachability reduction; it is conservative by design.

        Inputs
        ------
        ``max_generators`` is the maximum number of generator columns to keep
        in the returned zonotope. It must be at least ``n``.

        Output
        ------
        A conservative outer-approximating zonotope with at most
        ``max_generators`` columns.

        Example
        -------
        For a 2D zonotope with 10 generators and ``max_generators=4``, the
        method keeps the 2 largest generators and replaces the remaining 8 by
        a 2D axis-aligned box.
        """
        self._validate_reduction_target(max_generators)
        if self.p <= max_generators:
            return self.copy()
        retained_count = max_generators - self.n
        order = np.argsort(np.linalg.norm(self.generators, axis=0))[::-1]
        sorted_generators = self.generators[:, order]
        retained = sorted_generators[:, :retained_count]
        discarded = sorted_generators[:, retained_count:]
        box = np.diag(np.sum(np.abs(discarded), axis=1))
        return Zonotope(self.center, np.hstack((retained, box)))

    def reduce_order_pca(self) -> "Zonotope":
        """Return an optional PCA-oriented box outer approximation.

        Equations
        ---------
        ``G_rot = U.T @ G`` and ``r_rot = |G_rot| 1`` give
        ``G_pca = U @ diag(r_rot)``.

        This is a general utility. It is not the direct paper reduction; use
        ``reduce_order_outer_box`` in the occupancy pipeline.

        Inputs
        ------
        Uses the current zonotope generators.

        Output
        ------
        A PCA-oriented box zonotope with ``n`` generator columns.

        Example
        -------
        A long thin 2D zonotope is replaced by an oriented box aligned with its
        dominant generator covariance directions.
        """
        if self.p == 0:
            return self.copy()
        covariance_like = self.generators @ self.generators.T
        basis, _, _ = np.linalg.svd(covariance_like, full_matrices=True)
        rotated = basis.T @ self.generators
        radius = np.sum(np.abs(rotated), axis=1)
        return Zonotope(self.center, basis @ np.diag(radius))

    def reduce_order_pca_box(self, max_generators: int) -> "Zonotope":
        """Keep dominant generators and PCA-box only discarded generators.

        Inputs
        ------
        ``max_generators`` is the requested maximum generator count.

        Output
        ------
        A conservative zonotope that keeps the largest generator columns and
        replaces the discarded columns by a PCA-oriented box.

        Example
        -------
        For a 2D zonotope with 8 generators and ``max_generators=4``, this
        keeps 2 dominant generators and adds a 2-generator PCA box for the rest.
        """
        self._validate_reduction_target(max_generators)
        if self.p <= max_generators:
            return self.copy()
        retained_count = max_generators - self.n
        order = np.argsort(np.linalg.norm(self.generators, axis=0))[::-1]
        sorted_generators = self.generators[:, order]
        retained = sorted_generators[:, :retained_count]
        discarded = Zonotope(np.zeros((self.n, 1)), sorted_generators[:, retained_count:])
        oriented_box = discarded.reduce_order_pca()
        return Zonotope(self.center, np.hstack((retained, oriented_box.generators)))

    def reduce_order(self, method: ReductionMethod, max_generators: int) -> "Zonotope":
        """Dispatch to ``box``, ``pca``, or ``pca-box`` reduction.

        Inputs
        ------
        ``method`` is one of ``"box"``, ``"pca"``, or ``"pca-box"``.

        ``max_generators`` is passed to reduction methods that need a target
        generator count.

        Output
        ------
        A reduced outer-approximating zonotope.

        Example
        -------
        ``zonotope.reduce_order("box", 6)`` applies the paper-oriented box
        reduction with at most 6 generators.
        """
        self._validate_reduction_target(max_generators)
        if method == "box":
            return self.reduce_order_outer_box(max_generators)
        if method == "pca":
            return self.reduce_order_pca()
        if method == "pca-box":
            return self.reduce_order_pca_box(max_generators)
        raise ValueError("method must be one of: 'box', 'pca', 'pca-box'.")

    def reduction_zonotope(self, method_red: ReductionMethod, n_gen_red: int) -> "Zonotope":
        """Compatibility wrapper for the legacy reduction method.

        Inputs
        ------
        ``method_red`` is the reduction method and ``n_gen_red`` is the target
        generator count.

        Output
        ------
        The same result as ``reduce_order(method_red, n_gen_red)``.

        Example
        -------
        ``zonotope.reduction_zonotope("box", 5)`` is equivalent to
        ``zonotope.reduce_order("box", 5)``.
        """
        return self.reduce_order(method_red, n_gen_red)

    def dilate_box(self, radius: np.ndarray) -> "Zonotope":
        """Add an axis-aligned margin ``<0, diag(radius)>``.

        Equation
        ---------
        ``Z_dilated = Z (+) <0, diag(r)>`` for non-negative ``r``.

        Paper relationship
        -------------------
        This is the fourth and final occupancy simplification operation. The
        numerical radius is an application configuration parameter.

        Inputs
        ------
        ``radius`` is a non-negative vector with one value per dimension.

        Output
        ------
        A new zonotope equal to the Minkowski sum of the original zonotope and
        an axis-aligned box margin.

        Example
        -------
        ``Zonotope([[0.0], [0.0]], np.eye(2)).dilate_box([0.5, 0.25])`` expands
        the coordinate bounds from ``[-1, 1]^2`` to ``[-1.5, 1.5] x [-1.25, 1.25]``.
        """
        radius_array = np.asarray(radius, dtype=float).reshape(-1)
        if radius_array.shape != (self.n,):
            raise ValueError("radius must contain one value per dimension.")
        if not np.all(np.isfinite(radius_array)) or np.any(radius_array < 0):
            raise ValueError("Dilation radii must be finite and non-negative.")
        margin = Zonotope(np.zeros((self.n, 1)), np.diag(radius_array))
        return self + margin

    def contains_point(self, point: np.ndarray, tolerance: float = 1e-9) -> bool:
        """Test membership through ``G xi = point-c``, ``-1 <= xi <= 1``.

        This solves a zero-objective linear feasibility problem and is intended
        for experimental containment metrics, not set generation.

        Inputs
        ------
        ``point`` is a coordinate vector with exactly ``n`` values.

        ``tolerance`` expands the allowed coefficient interval slightly to
        handle numerical solver tolerance.

        Output
        ------
        ``True`` if the point is inside the zonotope, otherwise ``False``.

        Example
        -------
        For ``Zonotope([[0.0], [0.0]], np.eye(2))``, point ``[0.5, -0.5]`` is
        contained and point ``[2.0, 0.0]`` is not contained.
        """
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        point_array = np.asarray(point, dtype=float).reshape(-1, 1)
        if point_array.shape != (self.n, 1):
            raise ValueError(f"point must contain exactly {self.n} coordinates.")
        if not np.all(np.isfinite(point_array)):
            raise ValueError("point contains NaN or infinity.")
        if self.p == 0:
            return bool(np.linalg.norm(point_array - self.center) <= tolerance)
        result = linprog(
            c=np.zeros(self.p),
            A_eq=self.generators,
            b_eq=(point_array - self.center).ravel(),
            bounds=[(-1.0 - tolerance, 1.0 + tolerance)] * self.p,
            method="highs",
        )
        return bool(result.success)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample ``c + G xi`` with independent ``xi_j ~ Uniform[-1,1]``.

        Inputs
        ------
        ``rng`` is an optional NumPy random generator. If omitted, a new default
        generator is created.

        Output
        ------
        A sampled point with shape ``(n, 1)``.

        Example
        -------
        ``Zonotope([[0.0]], [[2.0]]).sample()`` returns one point in the
        interval ``[-2, 2]``.
        """
        random_generator = rng if rng is not None else np.random.default_rng()
        coefficients = random_generator.uniform(-1.0, 1.0, size=(self.p, 1))
        return self.center + self.generators @ coefficients

    def select_rand_sample(self) -> np.ndarray:
        """Compatibility alias for ``sample``.

        Output
        ------
        A sampled point with shape ``(n, 1)``.

        Example
        -------
        ``zonotope.select_rand_sample()`` is the same as ``zonotope.sample()``.
        """
        return self.sample()

    def v_cube(self, max_generators: int = 20) -> np.ndarray:
        """Return columns of ``{-1,+1}^p`` for exact vertex enumeration.

        Inputs
        ------
        ``max_generators`` limits enumeration because the number of sign
        combinations is ``2**p``.

        Output
        ------
        A matrix with shape ``(p, 2**p)`` containing all sign vectors.

        Example
        -------
        For ``p=2``, the columns are the four combinations of ``[-1, +1]``.
        """
        if self.p > max_generators:
            raise ValueError(f"Cannot enumerate 2^{self.p} sign combinations.")
        if self.p == 0:
            return np.empty((0, 1), dtype=float)
        return np.asarray(list(product((-1.0, 1.0), repeat=self.p)), dtype=float).T

    def find_vertices(self, max_generators: int = 20) -> np.ndarray:
        """Return convex-hull vertices for small sets as an ``ndarray``.

        Inputs
        ------
        ``max_generators`` limits exact enumeration of ``2**p`` sign vectors.

        Output
        ------
        An array whose rows are vertex coordinates.

        Example
        -------
        For ``Zonotope([[0.0]], [[2.0]])``, the returned vertices are ``[-2]``
        and ``[2]``.
        """
        if self.n == 1:
            lower, upper = self.bounds()
            return np.array([[lower[0, 0]], [upper[0, 0]]])
        candidates = (self.center + self.generators @ self.v_cube(max_generators)).T
        unique = np.unique(np.round(candidates, decimals=12), axis=0)
        if unique.shape[0] <= self.n:
            return unique
        try:
            hull = ConvexHull(unique)
        except QhullError:
            return unique
        return unique[hull.vertices]

    def vertices_2d(self, max_generators: int = 20) -> np.ndarray:
        """Return 2D convex-hull vertices.

        Inputs
        ------
        ``max_generators`` limits exact sign-combination enumeration.

        Output
        ------
        An array whose rows are ordered or hull-selected 2D vertex coordinates.

        Example
        -------
        ``Zonotope(np.zeros((2, 1)), np.eye(2)).vertices_2d()`` returns the
        four vertices of the unit square.
        """
        if self.n != 2:
            raise ValueError("vertices_2d requires a two-dimensional zonotope.")
        return self.find_vertices(max_generators)

    def area_2d(self) -> float:
        """Return ``4 * sum_{i<j} abs(det([g_i | g_j]))``.

        Inputs
        ------
        Uses the current two-dimensional zonotope generators.

        Output
        ------
        The exact 2D zonotope area as a float.

        Example
        -------
        ``Zonotope(np.zeros((2, 1)), np.eye(2)).area_2d()`` returns ``4.0``.
        """
        if self.n != 2:
            raise ValueError("area_2d requires a two-dimensional zonotope.")
        return 4.0 * sum(
            abs(float(np.linalg.det(self.generators[:, [i, j]])))
            for i, j in combinations(range(self.p), 2)
        )

    def volume(self) -> float:
        """Return ``2**n * sum_{|I|=n} abs(det(G_I))``.

        Inputs
        ------
        Uses the current ``n``-dimensional generator matrix.

        Output
        ------
        The exact zonotope volume as a float. If ``p < n``, the volume is zero.

        Example
        -------
        ``Zonotope(np.zeros((3, 1)), np.eye(3)).volume()`` returns ``8.0``.
        """
        if self.p < self.n:
            return 0.0
        total = sum(
            abs(float(np.linalg.det(self.generators[:, indices])))
            for indices in combinations(range(self.p), self.n)
        )
        return (2.0**self.n) * total

    def volume_zon(self) -> float:
        """Compatibility alias for ``volume``.

        Output
        ------
        The same scalar result as ``volume()``.

        Example
        -------
        ``zonotope.volume_zon()`` is equivalent to ``zonotope.volume()``.
        """
        return self.volume()

    def _require_same_dimension(self, other: "Zonotope", operation: str) -> None:
        if self.n != other.n:
            raise ValueError(f"{operation} requires equal dimensions.")

    def _validate_reduction_target(self, max_generators: int) -> None:
        if not isinstance(max_generators, (int, np.integer)):
            raise TypeError("max_generators must be an integer.")
        if max_generators < self.n:
            raise ValueError("max_generators must be at least the zonotope dimension.")


def cartesian_product_zonotope(first: Zonotope, second: Zonotope) -> Zonotope:
    """Return ``Z1 x Z2`` with stacked centers and block-diagonal generators.

    Inputs
    ------
    ``first`` and ``second`` are independent zonotopes.

    Output
    ------
    A new zonotope with center ``[c1; c2]`` and block-diagonal generator
    matrix ``diag_block(G1, G2)``.

    Example
    -------
    The Cartesian product of ``[-1, 2]`` and ``[3, 5]`` is a 2D box with
    coordinate bounds ``[-1, 2] x [3, 5]``.
    """
    if not isinstance(first, Zonotope) or not isinstance(second, Zonotope):
        raise TypeError("first and second must be Zonotope instances.")
    generators = np.block([
        [first.generators, np.zeros((first.n, second.p))],
        [np.zeros((second.n, first.p)), second.generators],
    ])
    return Zonotope(np.vstack((first.center, second.center)), generators)


def envbox_min_max(
    H: np.ndarray,
    scheduling_limits: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy scheduling helper; use ``Zonotope.bounds`` in new code.

    The important correction is ``sum(abs(H[i, :]))``. Generator cancellation
    is never allowed to shrink a safety interval.

    Inputs
    ------
    ``H`` is a generator-like matrix, ``scheduling_limits`` contains legacy
    bounds, and ``n`` is the expected row dimension.

    Output
    ------
    ``(minimum, maximum)`` coordinate arrays for the legacy envelope box.

    Example
    -------
    For new code, prefer ``Zonotope(center, H).bounds()`` because it directly
    returns the interval hull using the same ``sum(abs(H), axis=1)`` rule.
    """
    matrix = np.asarray(H, dtype=float)
    limits = np.asarray(scheduling_limits, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("H must be a two-dimensional matrix.")
    if matrix.shape[0] != n:
        raise ValueError("n must equal the number of rows in H.")
    radius = np.sum(np.abs(matrix), axis=1)

    if n == 3:
        minimum = -radius.copy()
        maximum = radius.copy()
        minimum[0] = limits[0, 0] + limits[0, 1] - radius[0]
    elif n == 2:
        minimum = np.array([-radius[0], limits[4, 0] + limits[4, 1] - radius[1]])
        maximum = radius.copy()
        minimum[0] = max(minimum[0], limits[3, 0])
        maximum[0] = min(maximum[0], limits[3, 1])
        minimum[1] = max(minimum[1], limits[4, 0])
        maximum[1] = min(maximum[1], limits[4, 1])
    else:
        raise ValueError("Only n=2 and n=3 are supported by this legacy helper.")
    return minimum, maximum


__all__ = ["ReductionMethod", "Zonotope", "cartesian_product_zonotope", "envbox_min_max"]
