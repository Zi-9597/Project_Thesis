"""Matrix interval set representation."""

from .intervals import Interval

import numpy as np
import numpy.typing as npt
from typing import Any

FloatArray = npt.NDArray[np.float64]

class MatrixInterval(Interval):
    """Two-dimensional interval matrix.

    ``MatrixInterval`` inherits the lower/upper storage and scalar interval
    arithmetic from :class:`Interval`, but requires two-dimensional bounds
    and adds operations needed by Equations (17)-(20):

    * construction from midpoint and radius matrices;
    * construction from explicit lower and upper matrices;
    * interval-matrix times real-matrix multiplication;
    * independent column concatenation; and
    * diamond-generator inclusion.

    Calling ``super().__init__`` initializes the inherited portion of the same
    object. It does not change the object type from ``MatrixInterval``.
    """

    def __init__(self, center: Any, delta: Any) -> None:
        """Construct ``[center-delta, center+delta]``.

        Require finite, same-shaped, two-dimensional matrices. Treat ``delta``
        as a magnitude by using its elementwise absolute value, matching the
        project convention for matrix deviations.

        Example: center ``[[2, 3], [4, 6]]`` and delta ``[[1, 1], [1, 2]]``
        produce lower ``[[1, 2], [3, 4]]`` and upper ``[[3, 4], [5, 8]]``.

        Parameters
        ----------
        center:
            A finite two-dimensional midpoint matrix.
        delta:
            A finite two-dimensional radius matrix with the same shape as
            ``center``. Its absolute value is used as the radius.

        Raises
        ------
        ValueError
            If either input is not two-dimensional, shapes differ, or values
            are not finite.
        """
        center_array = np.asarray(center, dtype=np.float64)
        delta_array = np.asarray(delta, dtype=np.float64)

        if center_array.ndim != 2 or delta_array.ndim != 2:
            raise ValueError(
                "center and delta must both be two-dimensional matrices."
            )

        if center_array.shape != delta_array.shape:
            raise ValueError(
                "center and delta must have the same shape; "
                f"received {center_array.shape} and {delta_array.shape}."
            )

        if not np.all(np.isfinite(center_array)):
            raise ValueError("center contains NaN or infinity.")

        if not np.all(np.isfinite(delta_array)):
            raise ValueError("delta contains NaN or infinity.")

        radius = np.abs(delta_array)
        super().__init__(center_array - radius, center_array + radius)
        
       
    @classmethod
    def from_bounds(cls, lower: Any, upper: Any) -> "MatrixInterval":
        """Create a matrix interval from lower and upper bound matrices.

        Each entry is constrained independently as
        ``lower[i, j] <= X[i, j] <= upper[i, j]``.

        Parameters
        ----------
        lower:
            Finite two-dimensional array of lower bounds.
        upper:
            Finite two-dimensional array of upper bounds with the same shape
            as ``lower`` and no value smaller than its corresponding lower
            bound.

        Returns
        -------
        MatrixInterval
            The interval represented by the supplied bounds.

        Raises
        ------
        ValueError
            If the inputs are not finite, two-dimensional, same-shaped
            arrays, or if a lower bound exceeds its upper bound.

        Examples
        --------
        >>> interval = MatrixInterval.from_bounds([[1.0, 2.0]], [[3.0, 6.0]])
        >>> interval.lower
        array([[1., 2.]])
        >>> interval.upper
        array([[3., 6.]])
        """
        lower_array = np.asarray(lower, dtype=np.float64)
        upper_array = np.asarray(upper, dtype=np.float64)

        if lower_array.ndim != 2 or upper_array.ndim != 2:
            raise ValueError("lower and upper must be two-dimensional matrices.")

        if lower_array.shape != upper_array.shape:
            raise ValueError(
                "lower and upper must have the same shape; "
                f"received {lower_array.shape} and {upper_array.shape}."
            )

        if not np.all(np.isfinite(lower_array)):
            raise ValueError("lower contains NaN or infinity.")

        if not np.all(np.isfinite(upper_array)):
            raise ValueError("upper contains NaN or infinity.")

        if np.any(lower_array > upper_array):
            raise ValueError("lower must not exceed upper elementwise.")

        center = (lower_array + upper_array) / 2.0
        radius = (upper_array - lower_array) / 2.0
        return cls(center, radius)

    @classmethod
    def point_matrix(cls, value: Any) -> "MatrixInterval":
        """Create a zero-width matrix interval ``[value, value]``.

        Parameters
        ----------
        value:
            A finite two-dimensional matrix.

        Returns
        -------
        MatrixInterval
            A matrix whose lower and upper bounds both equal ``value``.

        Example
        -------
        >>> MatrixInterval.point_matrix([[1.0, 2.0]]).radius
        array([[0., 0.]])
        """
        
        value_array : np.ndarray = np.asarray(value , dtype=float)
        delta_array : np.ndarray = np.zeros_like(value_array , dtype=float)
        
        return cls(center=value_array , delta = delta_array)

        
    @property
    def center_matrix(self) -> FloatArray:
        """Return the midpoint matrix ``mid([M])``.

        Returns
        -------
        FloatArray
            A copy of the elementwise midpoint matrix.
        """
        
        return self.midpoint.copy()

    @property
    def delta_matrix(self) -> FloatArray:
        """Return the radius matrix ``rad([M])``.

        Returns
        -------
        FloatArray
            A copy of the elementwise nonnegative radius matrix.
        """
        return self.radius.copy()
    def __matmul__(self, other: Any) -> "MatrixInterval":
        """Multiply an interval matrix by a real matrix.

        For ``[A]`` with shape ``(n, r)`` and real ``H`` with shape ``(r, p)``,
        compute the enclosure of ``[A] @ H`` using sign decomposition:

        ``H_plus = max(H, 0)``, ``H_minus = min(H, 0)``;

        ``lower = A_lower @ H_plus + A_upper @ H_minus``;

        ``upper = A_upper @ H_plus + A_lower @ H_minus``.

        This is the interval-matrix operation required by
        ``[A_k] H_x(k)`` and ``[B_k] H_u`` in Equation (17). Require a finite,
        two-dimensional right operand with compatible dimensions.

        Parameters
        ----------
        other:
            A finite real NumPy matrix ``H`` with shape ``(r, p)`` when this
            interval matrix has shape ``(n, r)``.

        Returns
        -------
        MatrixInterval
            The interval enclosure of ``[A] @ H`` with shape ``(n, p)``.

        Example
        -------
        >>> matrix = MatrixInterval.from_bounds([[1.0]], [[2.0]])
        >>> (matrix @ np.array([[3.0]])).lower
        array([[3.]])
        """
        
        
        other_matrix = np.asarray(other, dtype=np.float64)

        if other_matrix.ndim != 2:
            raise ValueError("other must be a two-dimensional matrix.")

        if not np.all(np.isfinite(other_matrix)):
            raise ValueError("other must contain only finite values.")

        if other_matrix.shape[0] != self.shape[1]:
            raise ValueError(
                "other has incompatible dimensions: "
                f"expected {self.shape[1]} rows, "
                f"got {other_matrix.shape[0]}."
            )

        h_minus = np.minimum(other_matrix, 0.0)
        h_plus = np.maximum(other_matrix, 0.0)

        lower = self._lower @ h_plus + self._upper @ h_minus
        upper = self._upper @ h_plus + self._lower @ h_minus

        return self.from_bounds(lower, upper)
            
        
        

    @classmethod
    def concatenate_columns(cls, *matrices: "MatrixInterval") -> "MatrixInterval":
        """Concatenate independent interval-generator column groups.

        For independent coefficient vectors, represent

        ``[M_x] beta_x + [M_u] beta_u``

        as

        ``[[M_x | M_u]] [beta_x; beta_u]``.

        Concatenate lower matrices separately from upper matrices. Require at
        least one ``MatrixInterval`` and identical row counts.

        Parameters
        ----------
        *matrices:
            One or more ``MatrixInterval`` objects with the same number of
            rows.

        Returns
        -------
        MatrixInterval
            A matrix interval formed by horizontal concatenation.

        Example
        -------
        >>> left = MatrixInterval.point_matrix([[1.0], [2.0]])
        >>> right = MatrixInterval.point_matrix([[3.0], [4.0]])
        >>> MatrixInterval.concatenate_columns(left, right).shape
        (2, 2)
        """

        if not matrices:
            raise ValueError("at least one MatrixInterval is required.")

        if not all(isinstance(matrix, MatrixInterval) for matrix in matrices):
            raise TypeError("all arguments must be MatrixInterval instances.")

        row_count = matrices[0].shape[0]
        if any(matrix.shape[0] != row_count for matrix in matrices[1:]):
            raise ValueError("all matrices must have the same number of rows.")

        lower_matrices = np.concatenate(
            [matrix._lower for matrix in matrices],
            axis=1,
        )
        upper_matrices = np.concatenate(
            [matrix._upper for matrix in matrices],
            axis=1,
        )

        return cls.from_bounds(lower_matrices, upper_matrices)

    def diamond_generators(
        self,
        *,
        remove_zero_columns: bool = True,
        zero_tolerance: float = 1e-14,
    ) -> FloatArray:
        """Return the real generator matrix of the diamond inclusion.

        Function signature
        ------------------
        The ``*`` in the function signature makes every parameter written
        after it a keyword-only argument.

        Therefore, the method should be called using explicit parameter
        names, for example:

            matrix.diamond_generators(
                remove_zero_columns=False,
                zero_tolerance=1e-12,
            )

        and not as:

            matrix.diamond_generators(False, 1e-12)

        Using keyword-only arguments improves readability because the meaning
        of values such as ``False`` and ``1e-12`` is immediately visible at
        the call site. It also reduces the risk of accidentally passing
        optional configuration parameters in the wrong order.

        Mathematical context
        --------------------
        Let ``[M]`` be an interval generator matrix represented by lower and
        upper matrices ``M_lower <= M <= M_upper`` elementwise. The interval
        represents a family of real matrices rather than one matrix.

        The diamond inclusion replaces this family with one real generator
        matrix ``J`` such that the resulting zonotope contains every zonotope
        generated by a matrix in ``[M]``. This construction corresponds to
        Theorem 3 of Alamo et al., *Guaranteed state estimation by zonotopes*.

        Mathematical construction
        -------------------------
        Compute

            M_mid = (M_lower + M_upper) / 2,
            R = (M_upper - M_lower) / 2,
            d_i = sum_j R_ij,
            D = diag(d_1, ..., d_n).

        The returned generator matrix is

            J = [M_mid | D].

        The midpoint columns preserve the nominal generators, while the
        diagonal columns bound the total interval uncertainty in each row.
        The zonotope center is computed separately and is not changed here.

        Parameters
        ----------
        remove_zero_columns:
            If true, remove generated columns whose Euclidean norm is at most
            ``zero_tolerance``.
        zero_tolerance:
            Finite, nonnegative threshold for identifying zero columns.

        Returns
        -------
        FloatArray
            Real generator matrix ``[M_mid | D]``. If ``[M]`` has shape
            ``(n, m)``, the matrix before optional removal has shape
            ``(n, m + n)``. The method returns generators only and does not
            construct a ``Zonotope`` object.

        Raises
        ------
        ValueError
            If ``zero_tolerance`` is negative, NaN, or infinite.

        Examples
        --------
        >>> matrix = MatrixInterval.from_bounds(
        ...     lower=[[1.0, -0.1], [0.4, 2.0]],
        ...     upper=[[1.2, 0.1], [0.6, 2.4]],
        ... )
        >>> matrix.diamond_generators(remove_zero_columns=False)
        array([[1.1, 0. , 0.2, 0. ],
               [0.5, 2.2, 0. , 0.3]])
        """
        
        
        midpoint = self.center_matrix
        row_radius = np.sum(self.delta_matrix, axis=1)
        diagonal_radius = np.diag(row_radius)

        generators = np.concatenate(
            (midpoint, diagonal_radius),
            axis=1,
        )

        if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
            raise ValueError("zero_tolerance must be finite and non-negative.")

        if remove_zero_columns:
            nonzero_columns = np.linalg.norm(generators, axis=0) > zero_tolerance
            generators = generators[:, nonzero_columns]

        return generators
        


# Preserve the spelling used by the original project while exposing the
# conventional class name for new code.
Matrix_Interval = MatrixInterval


__all__ = [
    "FloatArray",
    "Interval",
    "MatrixInterval",
    "Matrix_Interval",
]


__all__ = ["MatrixInterval", "Matrix_Interval"]
