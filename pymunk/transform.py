import math
from typing import NamedTuple

from .mat22 import Mat22
from .vec2d import Vec2d, VecLike


class Transform(NamedTuple):
    """Type used for 2x3 affine transforms.

    See wikipedia for details:
    http://en.wikipedia.org/wiki/Affine_transformation

    The properties map to the matrix in this way:

    = = ==
    a c tx
    b d ty
    0 0  1
    = = ==

    An instance can be created in this way:

        >>> Transform(1,2,3,4,5,6)
        Transform(a=1, b=2, c=3, d=4, tx=5, ty=6)

    Or overriding only some of the values (on a identity matrix):

        >>> Transform(b=3,ty=5)
        Transform(a=1, b=3, c=0, d=1, tx=0, ty=5)

    Or using one of the static methods like identity or translation (see each
    method for details).

    """

    a: float = 1
    b: float = 0
    c: float = 0
    d: float = 1
    tx: float = 0
    ty: float = 0

    @staticmethod
    def identity() -> "Transform":
        """The identity transform

        Example:

        >>> Transform.identity()
        Transform(a=1, b=0, c=0, d=1, tx=0, ty=0)

        Returns a Transform with this matrix:

        = = =
        1 0 0
        0 1 0
        0 0 1
        = = =

        """
        return Transform(1, 0, 0, 1, 0, 0)

    @staticmethod
    def translation(x: float, y: float) -> "Transform":
        """A translation transform

        Example to translate (move) by 3 on x and 5 in y axis:

        >>> Transform.translation(3, 5)
        Transform(a=1, b=0, c=0, d=1, tx=3, ty=5)

        Returns a Transform with this matrix:

        = = =
        1 0 x
        0 1 y
        0 0 1
        = = =

        """
        return Transform(tx=x, ty=y)

    # split into scale and scale_non-uniform
    @staticmethod
    def scaling(s: float) -> "Transform":
        """A scaling transform

        Example to scale 4x:

        >>> Transform.scaling(4)
        Transform(a=4, b=0, c=0, d=4, tx=0, ty=0)

        Returns a Transform with this matrix:

        = = =
        s 0 0
        0 s 0
        0 0 1
        = = =

        """
        return Transform(a=s, d=s)

    @staticmethod
    def rotation(t: float) -> "Transform":
        """A rotation transform

        Example to rotate by 1 rad:

        >>> Transform.rotation(1)
        Transform(a=0.5403023058681398, b=0.8414709848078965, c=-0.8414709848078965,
        d=0.5403023058681398, tx=0, ty=0)

        Returns a Transform with this matrix:

        ====== ======= =
        cos(t) -sin(t) 0
        sin(t) cos(t)  0
        0      0       1
        ====== ======= =

        """
        c = math.cos(t)
        s = math.sin(t)
        return Transform(a=c, b=s, c=-s, d=c)

    @staticmethod
    def projection(t, translation=(0, 0)):
        """
        Create a projection matrix.
        """
        return Transform.affine(Mat22.projection(t), translation)

    @staticmethod
    def affine(matrix=None, translation=(0, 0)):
        """
        Create transform from linear transformation encoded in matrix and
        translation.
        """
        a, b, c, d = Mat22.identity() if matrix is None else matrix
        tx, ty = translation
        return Transform(a, b, c, d, tx, ty)

    @staticmethod
    def similarity(*, scale=None, angle=None, angle_degrees=None, translation=(0, 0)):
        """
        Create affine transform from similarity transformations..
        """
        if angle is not None:
            mat = Mat22.rotation(angle)
        elif angle_degrees is not None:
            mat = Mat22.rotation(angle_degrees)
        else:
            mat = Mat22.identity()
        vec = Vec2d(*translation)

        if scale is not None:
            vec *= scale
            mat = Mat22.scale(scale) * mat

        return Transform.affine(mat, vec)

    def transform(self, vec: VecLike) -> Vec2d:
        """
        Return transformed vector by affine transform.
        """
        x, y = vec
        a, b, c, d, tx, ty = self
        return Vec2d(a * x + b * y + tx, c * x + d * y + ty)

    @property
    def matrix(self):
        """
        Linear component of the transform.
        """
        return Mat22(self.a, self.b, self.c, self.d)

    @property
    def vector(self):
        """
        Translation vector.
        """
        return Vec2d(self.tx, self.ty)

    def __mul__(self, other):
        if isinstance(other, Mat22):
            return Transform.affine(other * self.matrix, other.T * self.vector)
        elif isinstance(other, Transform):
            mat = self.matrix
            return Transform.affine(mat * other.matrix, self.vector + mat * other.vector)
        elif isinstance(other, (tuple, Vec2d)):
            return self.transform_vector(other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Mat22):
            return Transform.affine(other * self.matrix, other * self.vector)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, (Vec2d, tuple)):
            mat = self.matrix
            return Transform.affine(mat, self.vector + mat * other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (Vec2d, tuple)):
            return Transform.affine(self.matrix, self.vector + other)
        return NotImplemented

    def transform_vector(self, vec: VecLike):
        """
        Transform vector by affine transform.
        """
        return self.matrix.transform_vector(vec) + self.vector
