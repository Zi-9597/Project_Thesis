"""Interval arithmetic and interval-matrix operations.

This module is intentionally limited to the interval layer of the project.
It does not define or replace :class:`zonotope.Zonotope`.

The central representation is an elementwise closed interval::

    [X] = [X_lower, X_upper]

where ``X_lower`` and ``X_upper`` are NumPy arrays with identical shapes.
The same :class:`Interval` type can represent scalar intervals, vectors of
intervals, or matrices of intervals. :class:`MatrixInterval` specializes the
same representation for two-dimensional matrices used by the vehicle
occupancy propagation equations.

Paper relationships
-------------------
The operations correspond to the interval expressions used in:

* *Behavior-Aware Online Prediction of Obstacle Occupancy using Zonotopes*;
  especially the uncertain coefficients in Equations (18)-(19), interval
  generator propagation in Equation (17), and the diamond inclusion in
  Equation (20).
* Alamo, Bravo, and Camacho, *Guaranteed state estimation by zonotopes*;
  especially the natural interval operations and the midpoint-plus-diagonal
  enclosure from Theorem 3.

Implementation policy
---------------------
This file is currently an implementation scaffold. Public methods contain
their intended contracts, formulas, examples, and validation requirements,
but deliberately do not contain the implementation. Implement one method at
a time and use its docstring as the local specification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


class Interval:
    """Represent a closed scalar, vector, or array interval.

    Mathematical definition
    -----------------------
    For same-shaped arrays ``lower`` and ``upper`` satisfying

    ``lower <= upper`` element by element,

    this object represents every array ``X`` satisfying

    ``lower <= X <= upper`` element by element.

    Construction should validate matching shapes, finite values, and the
    lower-bound invariant. Bounds should be copied so callers cannot mutate
    the interval accidentally after construction.

    Examples
    --------
    ``Interval(2.0, 6.0)`` represents ``[2, 6]``.

    ``Interval([[1, 2]], [[3, 4]])`` represents a matrix with two interval
    entries: ``[[[1, 3], [2, 4]]]``.
    """

    __array_priority__ = 1000

    def __init__(self, lower: Any, upper: Any) -> None:
        """Validate and store the lower and upper bounds.

        Convert inputs with ``np.asarray(..., dtype=float)``. Scalar inputs
        should become zero-dimensional arrays, while vectors and matrices
        should preserve their dimensions.

        Raise ``ValueError`` when:

        * the bound shapes differ;
        * either bound contains NaN or infinity; or
        * any lower entry exceeds its corresponding upper entry.
        """
        
        lower_array: np.ndarray = np.asarray(lower, dtype=np.float64)
        upper_array: np.ndarray = np.asarray(upper, dtype=np.float64)

        if lower_array.shape != upper_array.shape:
            raise ValueError(
                "lower and upper bounds must have the same shape; "
                f"received {lower_array.shape} and {upper_array.shape}."
            )

        if not np.all(np.isfinite(lower_array)):
            raise ValueError("lower contains NaN or infinity.")

        if not np.all(np.isfinite(upper_array)):
            raise ValueError("upper contains NaN or infinity.")

        if np.any(lower_array > upper_array):
            raise ValueError(
                "some lower-bound values exceed their corresponding "
                "upper-bound values."
            )

        self._lower = lower_array.copy()
        self._upper = upper_array.copy()

    @classmethod
    def point(cls, value: Any) -> "Interval":
        """Create an interval ``[value, value]``.

        This is useful when combining an interval with a real scalar, vector,
        or matrix. For example, ``Interval.point(3.0)`` represents ``[3, 3]``.
        """

        value_array : np.ndarray = np.asarray(value , dtype=float)
        
        
        return cls(value_array , value_array)

    @classmethod
    def from_center_radius(cls, center: Any, radius: Any) -> "Interval":
        """Construct ``[center - radius, center + radius]``.

        Mathematical definition
        -----------------------
        Given midpoint ``c`` and nonnegative radius ``r``:

        ``[X] = [c - r, c + r]``.

        ``center`` and ``radius`` may be broadcast-compatible NumPy inputs.
        Reject non-finite values and negative radii.

        Example: ``Interval.from_center_radius(4.0, 2.0)`` represents
        ``[2, 6]``.
        """
        
        
        # Convert the inputs to floating-point NumPy arrays while preserving
        # scalar, vector, and matrix shapes.
        center_array: np.ndarray = np.asarray(center, dtype=float)
        radius_array: np.ndarray = np.asarray(radius, dtype=float)

        # This generic interval supports at most two dimensions. Higher
        # dimensions are rejected instead of being silently accepted.
        if center_array.ndim > 2 or radius_array.ndim > 2:
            raise ValueError(
                "center and radius must be scalars, vectors, or matrices."
            )

        # Broadcasting allows a scalar or compatible array radius to be used
        # with every corresponding entry of the center array.
        try:
            center_array, radius_array = np.broadcast_arrays(
                center_array,
                radius_array,
            )
        except ValueError as exc:
            raise ValueError(
                "center and radius must be broadcast-compatible."
            ) from exc

        # Bounds must be finite real values, and a radius cannot be negative.
        if not np.all(np.isfinite(center_array)):
            raise ValueError("center contains NaN or infinity.")

        if not np.all(np.isfinite(radius_array)):
            raise ValueError("radius contains NaN or infinity.")

        if np.any(radius_array < 0.0):
            raise ValueError("radius must be non-negative.")

        # Construct the symmetric interval [center - radius, center + radius].
        lower_bound = center_array - radius_array
        upper_bound = center_array + radius_array

        # Delegate final bound validation and defensive copying to __init__.
        return cls(lower_bound, upper_bound)
            
        
        
    @classmethod
    def from_affine_set(cls, center: Any, generators: Any) -> "Interval":
        """Return the coordinate hull of ``center + generators @ [-1, 1]^p``.

        For an affine set

        ``X = {c + G beta : beta_j in [-1, 1]}``,

        each coordinate has radius

        ``r_i = sum_j abs(G_ij)``.

        Therefore the enclosing interval is ``[c-r, c+r]``. This method is
        only a numerical bridge from existing center/generator arrays; it does
        not construct or depend on a Zonotope object.

        Normalize the center to a column shape ``(n, 1)`` and require the
        generator array to have shape ``(n, p)``.

        Example:

        ``center=[[0.2], [8.0]]`` and
        ``generators=[[0.02, -0.01], [0.2, 0.1]]`` produce
        ``[[0.17, 0.23], [7.7, 8.3]]`` as a column interval.
        """
        # Normalize the center to the project's standard column-vector shape.
        center_array = np.asarray(center, dtype=float).reshape(-1, 1)

        # Generators must form a matrix with shape (n, p): n coordinates and
        # p independent generator directions.
        generator_array = np.asarray(generators, dtype=float)

        if generator_array.ndim != 2:
            raise ValueError(
                "generators must be a two-dimensional matrix."
            )

        # Each generator row corresponds to one center coordinate.
        if center_array.shape[0] != generator_array.shape[0]:
            raise ValueError(
                "generator rows must match the center dimension; "
                f"received center {center_array.shape} and generators "
                f"{generator_array.shape}."
            )

        # For X = c + G beta with beta in [-1, 1]^p, the radius of coordinate
        # i is the sum of the absolute values in row i of G.
        coordinate_radius = np.sum(
            np.abs(generator_array),
            axis=1,
            keepdims=True,
        )

        # Construct [center - radius, center + radius]. The factory method also
        # validates finiteness and the nonnegative radius invariant.
        return cls.from_center_radius(
            center_array,
            coordinate_radius,
        )
        
        

    @property
    def lower(self) -> FloatArray:
        """Return a defensive copy of the lower-bound array."""
        return self._lower.copy()

    @property
    def upper(self) -> FloatArray:
        """Return a defensive copy of the upper-bound array."""
        return self._upper.copy()

    @property
    def midpoint(self) -> FloatArray:
        """Return ``(lower + upper) / 2`` element by element.

        The midpoint is the nominal value used by the diamond inclusion.
        """
        
        return (self._lower + self._upper)*0.5
    
    @property
    def diameter(self) -> FloatArray:
        """Return ``upper - lower`` element by element."""
        return self._upper - self._lower

    @property
    def radius(self) -> FloatArray:
        """Return ``(upper - lower) / 2`` element by element."""
        return self.diameter *0.5

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the common shape of the bound arrays."""
        return self._lower.shape

    @property
    def size(self) -> int:
        """Return the total number of interval entries."""
        return self._lower.size

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the interval arrays."""
        return self._lower.ndim

    def copy(self) -> "Interval":
        """Return an independent interval with identical bounds."""
        return Interval(self._lower , self._upper)

    def contains(self, value: Any, tolerance: float = 1e-9) -> bool:
        """Test elementwise membership of a scalar, vector, or matrix.

        A candidate is contained when every entry satisfies

        ``lower - tolerance <= value <= upper + tolerance``.

        Require the candidate to have exactly the interval shape. Reject
        negative, non-finite tolerances and non-finite candidate values.
        ``np.all`` must reduce all elementwise comparisons to one boolean.

        Example: ``Interval([1, 10], [3, 20]).contains([2, 15])`` is true,
        while ``contains([2, 25])`` is false.
        """
        
        
        
        value_array : np.ndarray = np.asarray(value , dtype=float)
        
        if value_array.shape != self.shape : 
            raise ValueError(
                "The value candidate has not the same shape as the interval sets"
            )
            
        # Zero tolerance is valid for exact containment; only negative values
        # would shrink the interval and are therefore rejected. Non-finite
        # tolerances cannot define a meaningful comparison margin.
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                "tolerance must be finite and non-negative"
            )

        # NaN and infinity are not valid real candidate values for membership.
        if not np.all(np.isfinite(value_array)):
            raise ValueError("value contains NaN or infinity.")
        
        lower_case = np.all(value_array >= self._lower - tolerance)
        upper_case = np.all(value_array <= self._upper + tolerance)
        
        return bool(lower_case and upper_case)

    def __getitem__(self, key: Any) -> "Interval":
        """Return the interval entry or subarray selected by ``key``.

        ``key`` has type ``Any`` because NumPy supports several indexing
        forms, including an integer, a tuple of indices, or a slice:

        ``interval[0]``, ``interval[1, 0]``, or ``interval[:, 0]``.

        The same key is applied to both lower and upper bounds.

        Example
        -------
        >>> interval = Interval(
        ...     [[1.0, 10.0], [3.0, 20.0]],
        ...     [[4.0, 15.0], [5.0, 25.0]],
        ... )
        >>> interval[0, 1].lower
        array(10.)
        >>> interval[0, 1].upper
        array(15.)
        """
        
        
        return Interval(self._lower[key] , self._upper[key])
        

    def __repr__(self) -> str:
        """Return the same readable representation as :meth:`__str__`."""
        return self.__str__()

    def __str__(self) -> str:
        """Return a readable representation of the interval bounds.

        Input
        -----
        The input is the current ``Interval`` object, using its stored
        ``self._lower`` and ``self._upper`` arrays.

        Output
        ------
        A string showing each interval entry as ``[lower, upper]``.

        The output format depends on the interval dimension:

        * scalar interval, ``ndim == 0``: one line;
        * vector interval, ``ndim == 1``: one interval per line;
        * matrix interval, ``ndim == 2``: one matrix row per line.

        Examples
        --------
        Scalar input::

            interval = Interval(2.0, 6.0)
            print(interval)

        Output::

            Interval([ 2.0 , 6.0])

        Vector input::

            interval = Interval([1.0, 10.0], [3.0, 20.0])
            print(interval)

        Output::

            Interval([
             [1.0 , 3.0]
             [10.0 , 20.0]
            ])

        Matrix input::

            interval = Interval(
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.5, 2.5], [3.5, 5.0]],
            )
            print(interval)

        Output::

            Interval([
            [  [ 1.0 , 1.5 ], [ 2.0 , 2.5 ]  ]
            [  [ 3.0 , 3.5 ], [ 4.0 , 5.0 ]  ]
            ])
        """
        
        
        if self.ndim == 0 : 
            
            return f"Interval([ {self._lower} , {self._upper}])"
        
        elif self.ndim == 1 : 
            
            row_list = [ f" [{lower_iter} , {upper_iter}]"
                        for lower_iter ,upper_iter in zip(self._lower , self._upper) ]
            
            
            return "Interval([ \n" + "\n".join(row_list) + "\n])"
            
        
        elif self.ndim == 2 : 
            
            row_list = []
            
            for lower_row , upper_row in zip(self._lower , self._upper) : 
                
                
                entries = [ f" [ {lower_iter} , {upper_iter} ]" for lower_iter , upper_iter in zip(lower_row , upper_row)]
                row_list.append("[ " + ",".join(entries) + "  ]")
                
            return "Interval([ \n" + "\n".join(row_list) + "\n])"
        else :
            
            raise ValueError("The shape to display is not possible it can be only scalar vector or matircx")

    @staticmethod
    def _coerce(value: Any) -> "Interval":
        """Convert an arithmetic operand into an ``Interval``.

        Parameters
        ----------
        value : Interval or array-like
            An existing ``Interval`` or a numeric scalar, vector, or matrix.

        Returns
        -------
        Interval
            The same interval when ``value`` is already an ``Interval``;
            otherwise the point interval ``[value, value]``.

        Example
        -------
        >>> Interval._coerce(2.0)
        Interval(lower=array(2.), upper=array(2.))

        This helper lets arithmetic accept both interval operands and exact
        numeric operands such as ``2.0`` or ``[1.0, 2.0]``.

        Design purpose
        --------------
        Centralizing this conversion keeps arithmetic methods such as
        ``__add__``, ``__sub__``, ``__mul__``, and ``__truediv__`` clean. They
        can operate on intervals directly without repeating the same
        ``isinstance`` check and point-interval construction.
        """
        
        if isinstance(value , Interval): 
            return value
        return Interval.point(value)
        

    def __add__(self, other: Any) -> "Interval":
        """Return the natural interval sum ``[a+c, b+d]`` elementwise.

        Parameters
        ----------
        other:
            ``Interval`` or array-like numeric operand. Non-interval operands
            are treated as exact values, equivalent to ``[other, other]``.

        Returns
        -------
        Interval
            The elementwise sum enclosure.

        Example
        -------
        >>> Interval(1.0, 3.0) + Interval(2.0, 4.0)
        Interval(lower=array(3.), upper=array(7.))
        """
        
        
        other_new = self._coerce(other)
        
        
        if self.shape != other_new.shape:
            raise ValueError(
                "interval operands must have the same shape; "
                f"received {self.shape} and {other_new.shape}."
            )
        
        return Interval(
            lower=self._lower + other_new._lower, 
            upper=self._upper + other_new._upper
        )
        
        
            

    def __radd__(self, other: Any) -> "Interval":
        """Return ``other + self`` for a left-hand real operand.

        Parameters
        ----------
        other:
            ``Interval`` or array-like numeric operand. Non-interval operands
            are treated as exact values, equivalent to ``[other, other]``.

        Returns
        -------
        Interval
            The elementwise interval sum.

        Example
        -------
        >>> 2.0 + Interval(1.0, 3.0)
        Interval(lower=array(3.), upper=array(5.))
        """
        return self + other

    def __neg__(self) -> "Interval":
        """Return the negated interval ``-[a,b] = [-b,-a]``.

        Returns
        -------
        Interval
            A new interval with negated and reversed bounds.

        Example
        -------
        >>> -Interval(2.0, 5.0)
        Interval(lower=array(-5.), upper=array(-2.))
        """
        return Interval(lower=(-1) * self._upper , upper=(-1)* self._lower)

    def __sub__(self, other: Any) -> "Interval":
        """Return the natural interval difference ``[a-d, b-c]``.

        Parameters
        ----------
        other:
            ``Interval`` or array-like numeric operand. Non-interval operands
            are treated as exact values, equivalent to ``[other, other]``.

        Returns
        -------
        Interval
            The elementwise difference enclosure.

        Example
        -------
        >>> Interval(5.0, 8.0) - Interval(1.0, 3.0)
        Interval(lower=array(2.), upper=array(7.))
        """
        
        other_new = self._coerce(other)
        
        return Interval(lower=self._lower-other_new._upper , upper=self._upper - other_new.lower)

    def __rsub__(self, other: Any) -> "Interval":
        """Return ``other - self`` for a left-hand real operand.

        Parameters
        ----------
        other:
            ``Interval`` or array-like numeric operand. Non-interval operands
            are treated as exact values, equivalent to ``[other, other]``.

        Returns
        -------
        Interval
            The enclosure of the reversed subtraction.

        Example
        -------
        >>> 10.0 - Interval(2.0, 4.0)
        Interval(lower=array(6.), upper=array(8.))
        """
        return -(self - other)

    def __mul__(self, other: Any) -> "Interval":
        """Return the natural elementwise interval product.

        For ``[a,b]`` and ``[c,d]``, evaluate ``ac``, ``ad``, ``bc``, and
        ``bd``. The result is their elementwise minimum and maximum.

         This supports expressions such as ``-Ts * velocity * theta.sin()``
         from the uncertain coefficients in the occupancy paper.

        Parameters
        ----------
        other:
            ``Interval`` or array-like numeric operand. Non-interval operands
            are treated as exact values, equivalent to ``[other, other]``.

        Returns
        -------
        Interval
            The elementwise enclosure obtained from the four endpoint
            products.

        Example
        -------
        >>> Interval(-2.0, 3.0) * Interval(4.0, 5.0)
        Interval(lower=array(-10.), upper=array(15.))
        """
        
        other_new = self._coerce(other)

        if self.shape != other_new.shape:
            raise ValueError(
                "interval operands must have the same shape; "
                f"received {self.shape} and {other_new.shape}."
            )

        product_candidate = np.stack(
            (
                self._lower * other_new._lower , 
                self._lower * other_new._upper , 
                self._upper * other_new._lower , 
                self._upper * other_new._upper
            ) , axis=0
        )
        
        return Interval(lower = np.min(product_candidate , axis = 0) , upper=np.max(product_candidate , axis = 0))

    def __rmul__(self, other: Any) -> "Interval":
        """Return ``other * self`` for a left-hand real operand.

        Parameters
        ----------
        other:
            ``Interval`` or array-like numeric operand. Non-interval operands
            are treated as exact values, equivalent to ``[other, other]``.

        Returns
        -------
        Interval
            The elementwise product enclosure.

        Example
        -------
        >>> 2.0 * Interval(1.0, 3.0)
        Interval(lower=array(2.), upper=array(6.))
        """
        return self * other

    def reciprocal(self) -> "Interval":
        """Return ``[1/b, 1/a]`` when zero is excluded.

        Reject every interval entry containing zero with ``ZeroDivisionError``.
        A finite compact reciprocal interval is not defined by this API when
         zero belongs to the denominator interval.

        Returns
        -------
        Interval
            The reciprocal interval ``[1/upper, 1/lower]``.

        Raises
        ------
        ZeroDivisionError
            If any entry contains zero.

        Example
        -------
        >>> Interval(2.0, 4.0).reciprocal()
        Interval(lower=array(0.25), upper=array(0.5))
        """
        
        contains_zero = (self._lower <= 0.0)&  (self._upper >=  0.0)
        if np.any(contains_zero) : 
            raise ZeroDivisionError("cannot compute the reciprocal of an interval containing zero.")
        
        return Interval(lower=1/self._upper , upper=1/self._lower)

    def __truediv__(self, other: Any) -> "Interval":
        """Return interval division as multiplication by a reciprocal.

        Parameters
        ----------
        other:
            ``Interval`` or array-like numeric denominator. Non-interval
            operands are treated as exact values. Every denominator entry
            must exclude zero.

        Returns
        -------
        Interval
            The enclosure of ``self / other``.

        Raises
        ------
        ZeroDivisionError
            If the denominator contains zero.

        Example
        -------
        >>> Interval(4.0, 8.0) / Interval(2.0, 4.0)
        Interval(lower=array(1.), upper=array(4.))
        """
        other_new = self._coerce(other)
        
        return self * other_new.reciprocal()

    def __rtruediv__(self, other: Any) -> "Interval":
        """Return ``other / self`` for a left-hand real operand.

        Parameters
        ----------
        other:
            ``Interval`` or array-like numeric numerator. The denominator is
            this interval and must exclude zero.

        Returns
        -------
        Interval
            The enclosure of the reversed division.

        Raises
        ------
        ZeroDivisionError
            If this interval contains zero.
        """
        other_new = self._coerce(other)
        return other_new * self.reciprocal()

    def sin(self) -> "Interval":
        """Compute an elementwise enclosure of the sine of this interval.

        For every scalar interval ``[a, b]``, this method computes an interval
        ``[y_lower, y_upper]`` satisfying

        ``sin(x) in [y_lower, y_upper]`` for every ``x in [a, b]``.

        Mathematical approach
        ---------------------
        The derivative of sine is

        ``d/dx sin(x) = cos(x)``.

        Therefore, internal extrema occur where ``cos(x) = 0``. The critical
        points are

        ``x_k = pi / 2 + k*pi,    k in Z``.

        At these points:

        ``sin(pi / 2 + k*pi) = (-1)^k``.

        Consequently:

        * even ``k`` gives the maximum value ``1``;
        * odd ``k`` gives the minimum value ``-1``.

        For each elementwise interval ``[a, b]``, the first and last valid
        critical-point indices are

        ``k_min = ceil((a - pi/2) / pi)`` and
        ``k_max = floor((b - pi/2) / pi)``.

        The critical-point range is nonempty exactly when ``k_min <= k_max``.
        If ``k_min + 1 <= k_max``, two consecutive indices are present. Two
        consecutive integers always contain one even and one odd index, so
        sine reaches both ``1`` and ``-1``. This avoids explicitly generating
        every integer between ``k_min`` and ``k_max``.

        The endpoint values ``sin(a)`` and ``sin(b)`` are always evaluated.
        Critical-point values then replace the corresponding endpoint-based
        bound when those critical points lie inside the interval.

        Returns
        -------
        Interval
            An interval whose bounds contain every possible elementwise sine
            value.

        Examples
        --------
        No internal critical point:

        >>> Interval(2.0, 4.0).sin()
        Interval(lower=array(-0.7568025), upper=array(0.90929743))

        One maximum critical point, with ``k = 0``:

        >>> Interval(1.0, 2.0).sin()
        Interval(lower=array(0.84147098), upper=array(1.))

        One minimum critical point, with ``k = 1``:

        >>> Interval(4.0, 5.0).sin()
        Interval(lower=array(-1.), upper=array(-0.7568025))

        Both a minimum and a maximum critical point:

        >>> Interval(4.0, 8.0).sin()
        Interval(lower=array(-1.), upper=array(1.))

        Notes
        -----
        Angles are measured in radians. The calculation is elementwise and
        supports scalar, vector, and matrix interval bounds. Each vector or
        matrix entry can have its own critical-point range.
        """
        sin_at_lower = np.sin(self._lower)
        sin_at_upper = np.sin(self._upper)

        # Endpoint values are always candidates for the result bounds.
        lower_sin = np.minimum(sin_at_lower, sin_at_upper)
        upper_sin = np.maximum(sin_at_lower, sin_at_upper)

        # Find the first and last k satisfying:
        # lower <= pi/2 + k*pi <= upper.
        k_min = np.ceil(
            (self._lower - np.pi / 2.0) / np.pi
        ).astype(np.int64)
        k_max = np.floor(
            (self._upper - np.pi / 2.0) / np.pi
        ).astype(np.int64)

        has_critical = k_min <= k_max
        k_min_is_even = (k_min % 2) == 0
        k_min_is_odd = (k_min % 2) != 0

        # Two consecutive valid indices necessarily contain both parities.
        has_next_critical = (k_min + 1) <= k_max

        contains_maximum = has_critical & (
            k_min_is_even | has_next_critical
        )
        contains_minimum = has_critical & (
            k_min_is_odd | has_next_critical
        )

        lower_sin = np.where(contains_minimum, -1.0, lower_sin)
        upper_sin = np.where(contains_maximum, 1.0, upper_sin)

        return Interval(
            lower=lower_sin,
            upper=upper_sin,
        )
        
        
        

    def cos(self) -> "Interval":
        """Compute an elementwise enclosure of the cosine of this interval.

        For every scalar interval ``[a, b]``, this method computes an interval
        ``[y_lower, y_upper]`` containing ``cos(x)`` for every ``x`` in
        ``[a, b]``.

        Mathematical approach
        ---------------------
        The derivative of cosine is ``d/dx cos(x) = -sin(x)``. Therefore,
        internal extrema occur where ``sin(x) = 0``. The critical points are

        ``x_k = k*pi,    k in Z``.

        At these points, ``cos(k*pi) = (-1)^k``. Even ``k`` gives the maximum
        value ``1`` and odd ``k`` gives the minimum value ``-1``.

        For each interval, the first and last critical-point indices are

        ``k_min = ceil(a / pi)`` and ``k_max = floor(b / pi)``.

        The range is nonempty when ``k_min <= k_max``. If ``k_min + 1 <=
        k_max``, two consecutive indices are present, so both parities and
        both extrema are present. Endpoint values are always included.

        Returns
        -------
        Interval
            An interval containing every possible elementwise cosine value.

        Examples
        --------
        One maximum critical point at ``x = 0``:

        >>> Interval(-0.5, 0.5).cos()
        Interval(lower=array(0.87758256), upper=array(1.))

        One minimum critical point at ``x = pi``:

        >>> Interval(2.0, 4.0).cos()
        Interval(lower=array(-1.), upper=array(-0.41614684))

        Both extrema:

        >>> Interval(0.0, 2.0 * np.pi).cos()
        Interval(lower=array(-1.), upper=array(1.))

        Notes
        -----
        Angles are measured in radians. The calculation supports scalar,
        vector, and matrix bounds elementwise.
        """
        cos_at_lower = np.cos(self._lower)
        cos_at_upper = np.cos(self._upper)

        # Endpoint values are always candidates for the result bounds.
        lower_cos = np.minimum(cos_at_lower, cos_at_upper)
        upper_cos = np.maximum(cos_at_lower, cos_at_upper)

        # Find the first and last k satisfying:
        # lower <= k*pi <= upper.
        k_min = np.ceil(
            self._lower / np.pi
        ).astype(np.int64)
        k_max = np.floor(
            self._upper / np.pi
        ).astype(np.int64)

        has_critical = k_min <= k_max
        k_min_is_even = (k_min % 2) == 0
        k_min_is_odd = (k_min % 2) != 0

        # Two consecutive valid indices necessarily contain both parities.
        has_next_critical = (k_min + 1) <= k_max

        contains_maximum = has_critical & (
            k_min_is_even | has_next_critical
        )
        contains_minimum = has_critical & (
            k_min_is_odd | has_next_critical
        )

        upper_cos = np.where(contains_maximum, 1.0, upper_cos)
        lower_cos = np.where(contains_minimum, -1.0, lower_cos)

        return Interval(
            lower=lower_cos,
            upper=upper_cos,
        )
        
