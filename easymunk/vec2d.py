# ----------------------------------------------------------------------------
# pymunk
# Copyright (c) 2007-2020 Victor Blomqvist, 2021 Fábio Macêdo Mendes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------
"""
This module contain the Vec2d class that is used in all of pymunk when a
vector is needed.

The Vec2d class is used almost everywhere in easymunk for 2d coordinates and
vectors, for example to define gravity vector in a space. However, pymunk is
smart enough to convert tuples or tuple like objects to Vec2ds so you usually
do not need to explicitly do conversions if you happen to have a tuple::

    >>> import easymunk as mk
    >>> from easymunk import Vec2d
    >>> space = mk.Space()
    >>> space.gravity
    Vec2d(0.0, 0.0)

    >>> space.gravity = 3,5
    >>> space.gravity
    Vec2d(3.0, 5.0)

More examples::

    >>> v = Vec2d(3, 4)
    >>> 2 * v
    Vec2d(6, 8)
    >>> v.length
    5.0
"""
__docformat__ = "reStructuredText"

import numbers
from typing import NamedTuple, Tuple, Any

from .math import sqrt, atan2, cos, sin, radians, degrees

__all__ = ["Vec2d", "VecLike", "vec2d_from_cffi"]

VecLike = Tuple[float, float]


class Vec2d(NamedTuple):
    """2d vector class, supports vector and scalar operators, and also
    provides some high level functions.
    """

    x: float
    y: float

    @property
    def int_tuple(self) -> Tuple[int, int]:
        """The x and y values of this vector as a tuple of ints.
        Uses round() to round to closest int.

        >>> Vec2d(0.9, 2.4).int_tuple
        (1, 2)
        """
        return round(self.x), round(self.y)

    @staticmethod
    def zero() -> "Vec2d":
        """A vector of zero length.

        >>> Vec2d.zero()
        Vec2d(0, 0)
        """
        return Vec2d(0, 0)

    @staticmethod
    def unit() -> "Vec2d":
        """A unit vector pointing up

        >>> Vec2d.unit()
        Vec2d(0, 1)
        """
        return Vec2d(0, 1)

    @staticmethod
    def ones() -> "Vec2d":
        """A vector where both x and y is 1

        >>> Vec2d.ones()
        Vec2d(1, 1)
        """
        return Vec2d(1, 1)

    @property
    def length_sqr(self) -> float:
        """Squared length of vector.

        If the squared length is enough it is more efficient to use this method
        instead of access .length and then do a x**2.

        >>> v = Vec2d(3, 4)
        >>> v.length_sqr == v.length ** 2 == 25.0
        True
        """
        return self.x ** 2 + self.y ** 2

    @property
    def length(self) -> float:
        """Length of vector.

        >>> Vec2d(3, 4).length
        5.0
        """
        return sqrt(self.x ** 2 + self.y ** 2)

    @property
    def angle(self) -> float:
        """The angle (in degrees) of the vector"""
        if self.length_sqr == 0:
            return 0
        return atan2(self.y, self.x)

    @property
    def angle_radians(self) -> float:
        """The angle (in radians) of the vector"""
        return radians(self.angle)

    # String representation (for debugging)
    def __repr__(self) -> str:
        return "Vec2d(%r, %r)" % (self.x, self.y)

    # Addition
    def __add__(self, other: VecLike) -> "Vec2d":  # type: ignore
        """Add a Vec2d with another Vec2d or Tuple of size 2

        >>> Vec2d(3,4) + Vec2d(1,2)
        Vec2d(4, 6)
        >>> Vec2d(3,4) + (1,2)
        Vec2d(4, 6)
        """
        try:
            u, v = other
        except (IndexError, TypeError):
            return NotImplemented
        else:
            return Vec2d(self.x + u, self.y + v)

    def __radd__(self, other: VecLike) -> "Vec2d":
        return self.__add__(other)

    # Subtraction
    def __sub__(self, other: VecLike) -> "Vec2d":
        """Subtract a Vec2d with another Vec2d or Tuple of size 2

        >>> Vec2d(3,4) - Vec2d(1,2)
        Vec2d(2, 2)
        >>> Vec2d(3,4) - (1,2)
        Vec2d(2, 2)
        """
        try:
            u, v = other
        except (IndexError, TypeError):
            return NotImplemented
        else:
            return Vec2d(self.x - u, self.y - v)

    def __rsub__(self, other: VecLike) -> "Vec2d":
        """Subtract a Tuple of size 2 with a Vec2d

        >>> (1,2) - Vec2d(3,4)
        Vec2d(-2, -2)
        """
        try:
            u, v = other
        except (IndexError, TypeError):
            return NotImplemented
        else:
            return Vec2d(u - self.x, v - self.y)

    # Multiplication
    def __mul__(self, other: float) -> "Vec2d":
        """Multiply with a float

        >>> Vec2d(3,6) * 2.5
        Vec2d(7.5, 15.0)
        """
        if other.__class__ is float or isinstance(other, numbers.Real):
            return Vec2d(self.x * other, self.y * other)
        return NotImplemented

    def __rmul__(self, other: float) -> "Vec2d":
        return self.__mul__(other)

    # Division
    def __floordiv__(self, other: float) -> "Vec2d":
        """Floor division by a float (also known as integer division)

        >>> Vec2d(3,6) // 2.0
        Vec2d(1.0, 3.0)
        """
        if other.__class__ is float or isinstance(other, numbers.Real):
            return Vec2d(self.x // other, self.y // other)
        return NotImplemented

    def __truediv__(self, other: float) -> "Vec2d":
        """Division by a float

        >>> Vec2d(3,6) / 2.0
        Vec2d(1.5, 3.0)
        """
        if other.__class__ is float or isinstance(other, numbers.Real):
            return Vec2d(self.x / other, self.y / other)
        return NotImplemented

    # Unary operations
    def __neg__(self) -> "Vec2d":
        """Return the negated version of the Vec2d

        >>> -Vec2d(1,-2)
        Vec2d(-1, 2)
        """
        return Vec2d(-self.x, -self.y)

    def __pos__(self) -> "Vec2d":
        """Return the unary pos of the Vec2d.

        >>> +Vec2d(1,-2)
        Vec2d(1, -2)
        """
        return Vec2d(+self.x, +self.y)

    def __abs__(self) -> float:
        """Return the length of the Vec2d

        >>> abs(Vec2d(3,4))
        5.0
        """
        return self.length

    def scale_to_length(self, length: float) -> "Vec2d":
        """Return a copy of this vector scaled to the given length.

        >>> Vec2d(10, 20).scale_to_length(20)
        Vec2d(8.94427190999916, 17.88854381999832)
        """
        old_length = self.length
        return Vec2d(self.x * length / old_length, self.y * length / old_length)

    def rotated(self, angle: float) -> "Vec2d":
        """Create and return a new vector by rotating this vector by
        angle (in degrees)."""
        cos_ = cos(angle)
        sin_ = sin(angle)
        x = self.x * cos_ - self.y * sin_
        y = self.x * sin_ + self.y * cos_
        return Vec2d(x, y)

    def rotated_radians(self, angle: float) -> "Vec2d":
        """Create and return a new vector by rotating this vector by
        angle (in radians)."""
        return self.rotated(degrees(angle))

    def angle_between(self, other: VecLike) -> float:
        """Get the angle between the vector and the other in degrees."""
        u, v = other
        cross = self.x * v - self.y * u
        dot = self.x * u + self.y * v
        return atan2(cross, dot)

    def angle_radians_between(self, other: VecLike) -> float:
        """Get the angle between the vector and the other in radians."""
        return radians(self.angle_between(other))

    def normalized(self) -> "Vec2d":
        """Get a normalized copy of the vector
        Note: This function will return 0 if the length of the vector is 0.

        :return: A normalized vector
        """
        length = self.length
        if length != 0:
            return self / length
        return Vec2d(0, 0)

    def normalized_and_length(self) -> Tuple["Vec2d", float]:
        """Normalize the vector and return its length before the normalization

        :return: The length before the normalization
        """
        length = self.length
        if length != 0:
            return self / length, length
        return Vec2d(0, 0), 0

    def perpendicular(self) -> "Vec2d":
        return Vec2d(-self.y, self.x)

    def perpendicular_normal(self) -> "Vec2d":
        length = self.length
        if length != 0:
            return Vec2d(-self.y / length, self.x / length)
        return Vec2d(self.x, self.y)

    def dot(self, other: VecLike) -> float:
        """The dot product between the vector and other vector
            v1.dot(v2) -> v1.x*v2.x + v1.y*v2.y

        :return: The dot product
        """
        u, v = other
        return float(self.x * u + self.y * v)

    def distance(self, other: VecLike) -> float:
        """The distance between the vector and other vector."""
        u, v = other
        return sqrt((self.x - u) ** 2 + (self.y - v) ** 2)

    def distance_sqr(self, other: VecLike) -> float:
        """The squared distance between the vector and other vector.

        It is more efficient to use this method than to call get_distance()
        first and then do a sqrt() on the result.
        """
        u, v = other
        return (self.x - u) ** 2 + (self.y - v) ** 2

    def projection(self, other: VecLike) -> "Vec2d":
        """Project this vector on top of other vector"""

        u, v = other
        other_length_sqrd = u * u + v * v
        if other_length_sqrd == 0.0:
            return Vec2d(0, 0)
        projected_length_times_other_length = self.dot(other)
        new_length = projected_length_times_other_length / other_length_sqrd
        return Vec2d(u * new_length, v * new_length)

    def cross(self, other: VecLike) -> float:
        """The cross product between the vector and other vector
        v1.cross(v2) -> v1.x*v2.y - v1.y*v2.x
        """
        u, v = other
        return self.x * v - self.y * u

    def interpolate_to(self, other: VecLike, ratio: float = 0.5) -> "Vec2d":
        """Interpolate with other vector.

        The "ratio" parameter determines the weight of self and other. If ratio=0,
        returns self and ratio=1 returns other. Intermediate values produce
        intermediate vectors.
        """
        u, v = other
        return Vec2d(self.x + (u - self.x) * ratio, self.y + (v - self.y) * ratio)

    def convert_to_basis(self, x_vector: VecLike, y_vector: VecLike) -> "Vec2d":
        return Vec2d(
            self.dot(x_vector) / Vec2d(*x_vector).length_sqr,
            self.dot(y_vector) / Vec2d(*y_vector).length_sqr,
        )

    # Extra functions, mainly for chipmunk
    def cpvrotate(self, other: VecLike) -> "Vec2d":
        """Uses complex multiplication to rotate this vector by the other. """
        u, v = other
        return Vec2d(self.x * u - self.y * v, self.x * v + self.y * u)

    def cpvunrotate(self, other: VecLike) -> "Vec2d":
        """The inverse of cpvrotate"""
        u, v = other
        return Vec2d(self.x * u + self.y * v, self.y * u - self.x * v)


def vec2d_from_cffi(cffi: Any) -> Vec2d:
    """
    Creates Vec2d from cffi cpVect.
    """
    return Vec2d(cffi.x, cffi.y)
