"""Matrix zonotopes with one independent scalar per generator matrix."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
import numpy.typing as npt

from .matrix_interval import MatrixInterval
from .zonotope import ReductionMethod, Zonotope

FloatArray = npt.NDArray[np.float64]


class MatrixZonotope:
    """Represent ``C + sum(beta_i G_i)`` for ``beta_i in [-1, 1]``."""

    def __init__(self, center: Any, generators: Any) -> None:
        center_array = np.asarray(center, dtype=np.float64)
        generator_array = np.asarray(generators, dtype=np.float64)

        # MIGRATION: The legacy file used assert-based validation and retained caller
        # arrays by reference. Explicit exceptions and copies preserve the same set
        # representation while keeping validation active in optimized Python runs.
        if center_array.ndim != 2:
            raise ValueError("center must be a two-dimensional matrix.")
        if generator_array.ndim != 3:
            raise ValueError("generators must have shape (m, n, p).")
        if generator_array.shape[1:] != center_array.shape:
            raise ValueError(
                "Each generator must have the same shape as center; "
                f"received {generator_array.shape[1:]} and {center_array.shape}."
            )
        if not np.all(np.isfinite(center_array)):
            raise ValueError("center contains NaN or infinity.")
        if not np.all(np.isfinite(generator_array)):
            raise ValueError("generators contain NaN or infinity.")

        self._center = center_array.copy()
        self._generators = generator_array.copy()

    @property
    def center(self) -> FloatArray:
        """Return a copy of the center matrix."""
        # MIGRATION: The legacy property exposed its internal array directly.
        # Returning a copy prevents callers from changing the represented set.
        return self._center.copy()

    @property
    def generators(self) -> FloatArray:
        """Return a copy of the generator-matrix batch."""
        # MIGRATION: The legacy property exposed its generator storage directly.
        # A defensive copy preserves the same values without external mutation.
        return self._generators.copy()

    @property
    def shape(self) -> tuple[int, int]:
        """Return the matrix shape ``(n, p)``."""
        return self._center.shape

    @property
    def num_generators(self) -> int:
        """Return the number of independent generator matrices."""
        return self._generators.shape[0]

    @property
    def Z(self) -> FloatArray:
        """Return the stack ``[C, G1, ..., Gm]``."""
        # MIGRATION: The legacy file cached [C, G1, ..., Gm]. Computing it on demand
        # preserves its ordering and prevents stale data if internals ever change.
        return np.concatenate((self._center[None, :, :], self._generators), axis=0)

    def copy(self) -> "MatrixZonotope":
        """Return an independent matrix zonotope representing the same set."""
        return MatrixZonotope(self._center.copy(), self._generators.copy())

    def __repr__(self) -> str:
        """Return an unambiguous constructor-style representation."""
        # MIGRATION: The legacy class defined only a human-readable __str__.
        # This representation makes the conventional class and inputs explicit.
        return f"MatrixZonotope(center={self._center!r}, generators={self._generators!r})"

    def __str__(self) -> str:
        """Return a readable center and generator summary."""
        return f"Center:\n{self._center}\n\nGenerators:\n{self._generators}"

    def _require_same_shape(self, other: "MatrixZonotope", operation: str) -> None:
        if self.shape != other.shape:
            raise ValueError(f"{operation} requires equal matrix shapes.")

    def _real_matrix(self, value: Any) -> FloatArray:
        try:
            matrix = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError("operand must be a finite real matrix.") from error
        if matrix.ndim != 2 or matrix.shape != self.shape:
            raise ValueError(f"operand must have shape {self.shape}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("operand contains NaN or infinity.")
        return matrix

    def _embedded_zonotope(self, zonotope: Zonotope) -> tuple[FloatArray, FloatArray]:
        if zonotope.n != self.shape[0]:
            raise ValueError("Zonotope dimension must equal the matrix row count.")
        if self.shape[1] == 0:
            raise ValueError("A Zonotope cannot be embedded in a matrix with zero columns.")

        # MIGRATION: Preserve the legacy first-column embedding exactly; this is a
        # compatibility convention, not a general matrix/vector broadcasting rule.
        center = np.zeros(self.shape, dtype=np.float64)
        center[:, 0] = zonotope.center[:, 0]
        generators = np.zeros((zonotope.p, *self.shape), dtype=np.float64)
        generators[:, :, 0] = zonotope.generators.T
        return center, generators

    def __add__(self, other: Any) -> "MatrixZonotope":
        if isinstance(other, MatrixZonotope):
            self._require_same_shape(other, "Matrix-zonotope addition")
            return MatrixZonotope(
                self._center + other._center,
                np.concatenate((self._generators, other._generators), axis=0),
            )
        if isinstance(other, Zonotope):
            center, generators = self._embedded_zonotope(other)
            return MatrixZonotope(
                self._center + center,
                np.concatenate((self._generators, generators), axis=0),
            )
        try:
            matrix = self._real_matrix(other)
        except TypeError:
            return NotImplemented
        return MatrixZonotope(self._center + matrix, self._generators)

    def __radd__(self, other: Any) -> "MatrixZonotope":
        # MIGRATION: The legacy class supported only matrix_zonotope + matrix.
        # Reversed addition represents the same Minkowski translation.
        return self + other

    def __sub__(self, other: Any) -> "MatrixZonotope":
        if isinstance(other, MatrixZonotope):
            self._require_same_shape(other, "Matrix-zonotope subtraction")
            return MatrixZonotope(
                self._center - other._center,
                np.concatenate((self._generators, -other._generators), axis=0),
            )
        if isinstance(other, Zonotope):
            center, generators = self._embedded_zonotope(other)
            return MatrixZonotope(
                self._center - center,
                np.concatenate((self._generators, -generators), axis=0),
            )
        try:
            matrix = self._real_matrix(other)
        except TypeError:
            return NotImplemented
        return MatrixZonotope(self._center - matrix, self._generators)

    def __rsub__(self, other: Any) -> "MatrixZonotope":
        if isinstance(other, Zonotope):
            return NotImplemented
        try:
            matrix = self._real_matrix(other)
        except TypeError:
            return NotImplemented
        # MIGRATION: The legacy class omitted matrix - matrix_zonotope. Negating
        # every generator gives the natural reflected set without changing its form.
        return MatrixZonotope(matrix - self._center, -self._generators)

    def __mul__(self, alpha: Any) -> "MatrixZonotope":
        if not isinstance(alpha, (Real, np.number)) or np.ndim(alpha) != 0:
            return NotImplemented
        scalar = float(alpha)
        if not np.isfinite(scalar):
            raise ValueError("alpha must be finite.")
        return MatrixZonotope(scalar * self._center, scalar * self._generators)

    def __rmul__(self, alpha: Any) -> "MatrixZonotope":
        # MIGRATION: The legacy class supported only M * alpha. Scalar
        # multiplication is commutative, so alpha * M has identical behavior.
        return self * alpha

    def linear_mapping(self, T: Any) -> "MatrixZonotope":
        """Apply the right mapping ``M -> M @ T``."""
        # MIGRATION: The legacy implementation validated only compatibility with
        # an assert. Explicit dimension and finiteness checks preserve the mapping.
        mapping = np.asarray(T, dtype=np.float64)
        if mapping.ndim != 2:
            raise ValueError("T must be a two-dimensional matrix.")
        if mapping.shape[0] != self.shape[1]:
            raise ValueError("T has an incompatible input dimension.")
        if not np.all(np.isfinite(mapping)):
            raise ValueError("T contains NaN or infinity.")
        return MatrixZonotope(self._center @ mapping, self._generators @ mapping)

    def zonotope_product(self, operand: Zonotope) -> Zonotope:
        """Return the legacy independent-term outer approximation of ``M @ Z``.

        Bilinear terms share coefficients in the exact product. Treating every
        pairwise term as independent therefore yields an outer approximation.
        """
        if not isinstance(operand, Zonotope):
            raise TypeError("operand must be a Zonotope.")
        if self.shape[1] != operand.n:
            raise ValueError("Matrix columns must equal the Zonotope dimension.")

        # MIGRATION: Preserve the legacy pairwise matrix-zonotope/zonotope product
        # pattern, but construct or document the term ordering explicitly.
        center = self._center @ operand.center
        generators: list[FloatArray] = []
        generators.extend(generator @ operand.center for generator in self._generators)
        generators.extend(self._center @ operand.generators[:, [j]] for j in range(operand.p))
        generators.extend(
            generator @ operand.generators[:, [j]]
            for generator in self._generators
            for j in range(operand.p)
        )
        generator_matrix = (
            np.hstack(generators)
            if generators
            else np.empty((self.shape[0], 0), dtype=np.float64)
        )
        return Zonotope(center=center, Gen=generator_matrix)

    def to_zonotope(self) -> Zonotope:
        """Flatten matrices in row-major order into an ordinary zonotope."""
        center = self._center.reshape(-1, 1)
        # MIGRATION: Reshaping the generator batch replaces np.hstack(list), which
        # failed for singleton matrix zonotopes while preserving row-major flattening.
        # The flattened dimension is explicit because NumPy cannot infer ``-1``
        # from an empty generator batch with shape ``(0, n, p)``.
        flattened_dimension = self.shape[0] * self.shape[1]
        generators = self._generators.reshape(
            self.num_generators,
            flattened_dimension,
        ).T
        return Zonotope(center=center, Gen=generators)

    def Mat2Zon(self) -> Zonotope:
        """Compatibility alias for :meth:`to_zonotope`."""
        return self.to_zonotope()

    def approximation_zonotope(self) -> "MatrixZonotope":
        """Return the legacy one-generator absolute-sum approximation.

        One scalar controls the complete radius matrix, so this is generally
        smaller than the independently varying elementwise interval hull.
        """
        # MIGRATION: This intentionally preserves the legacy single-generator
        # construction; it is distinct from the elementwise interval hull below.
        delta = np.sum(np.abs(self._generators), axis=0)
        return MatrixZonotope(self._center, delta[None, :, :])

    def reduce_order(
        self,
        nb_generators: int,
        method: ReductionMethod = "box",
    ) -> "MatrixZonotope":
        """Reduce after flattening; the target must be at least ``n * p``."""
        # MIGRATION: The legacy method printed and silently returned the unreduced set
        # on any exception. Propagating the error prevents unnoticed reduction failure.
        reduced = self.to_zonotope().reduce_order(method, nb_generators)
        center = reduced.center.reshape(self.shape)
        generators = reduced.generators.T.reshape(reduced.p, *self.shape)
        return MatrixZonotope(center, generators)

    def interval_matrix(self) -> MatrixInterval:
        """Return the exact elementwise interval hull."""
        # MIGRATION: The legacy formula is preserved, but the result now uses the
        # package's current MatrixInterval implementation and import path.
        delta = np.sum(np.abs(self._generators), axis=0)
        return MatrixInterval(self._center, delta)

    def find_vertices(self) -> FloatArray:
        """Return flattened-space convex-hull vertices reshaped as matrices."""
        vertices = self.to_zonotope().find_vertices()
        # MIGRATION: The legacy reshape used vertices[0] as a dimension. Use the
        # number of vertex rows while preserving matrix-entry ordering.
        return vertices.reshape(vertices.shape[0], *self.shape)


Matrix_Zonotope = MatrixZonotope


def concatenate_zonotope(zon: Zonotope, T: int) -> MatrixZonotope:
    """Create ``T`` independent column copies of an ordinary zonotope."""
    if not isinstance(zon, Zonotope):
        raise TypeError("zon must be a Zonotope.")
    if not isinstance(T, (int, np.integer)):
        raise TypeError("T must be an integer.")
    if T <= 0:
        raise ValueError("T must be positive.")

    center = np.repeat(zon.center, T, axis=1)
    generators = np.zeros((T * zon.p, zon.n, T), dtype=np.float64)
    for generator_index in range(zon.p):
        for column_index in range(T):
            generators[column_index + T * generator_index, :, column_index] = (
                zon.generators[:, generator_index]
            )

    # MIGRATION: Remove diagnostic prints; the returned center and generator
    # layout remains identical to the legacy construction.
    return MatrixZonotope(center, generators)


__all__ = ["MatrixZonotope", "Matrix_Zonotope", "concatenate_zonotope"]
