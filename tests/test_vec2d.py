import math
import pickle

import pytest
from pytest import approx

from easymunk.vec2d import Vec2d


class TestVec2d:
    def testCreationAndAccess(self) -> None:
        v = Vec2d(*(0, 0))
        assert v.x == 0
        assert v[0] == 0
        assert v.y == 0
        assert v[1] == 0

        v = Vec2d(3, 5)
        assert v.x == 3
        assert v[0] == 3
        assert v.y == 5
        assert v[1] == 5

        v = Vec2d(111, 222)
        assert v.x == 111 and v.y == 222
        with pytest.raises(AttributeError):
            v.x = 333  # type: ignore
        with pytest.raises(TypeError):
            v[1] = 444  # type: ignore

        v = Vec2d(3, 5)
        assert len(v) == 2
        assert list(v) == [3, 5]
        assert tuple(v) == (3, 5)

    def testMath(self) -> None:
        v = Vec2d(111, 222)
        assert v + Vec2d(1, 2) == Vec2d(112, 224)
        assert v + (1, 2) == Vec2d(112, 224)
        assert (1, 2) + v == Vec2d(112, 224)

        assert v - Vec2d(1, 2) == Vec2d(110, 220)
        assert v - (1, 2) == Vec2d(110, 220)
        assert (1, 2) - v == Vec2d(-110, -220)

        assert v * 3 == Vec2d(333, 666)
        assert 3 * v == Vec2d(333, 666)

        assert v / 2 == Vec2d(55.5, 111)
        assert v // 2 == Vec2d(55, 111)

    def testUnary(self) -> None:
        v = Vec2d(111, 222)
        assert +v == v
        assert -v == Vec2d(-111, -222)
        assert abs(v) == approx(248.20354550247666)

    def testLength(self) -> None:
        v = Vec2d(3, 4)
        assert v.length == 5
        assert v.length_sqr == 25

        normalized, length = v.normalized_and_length()
        assert normalized == Vec2d(0.6, 0.8)
        assert length == 5

        normalized, length = Vec2d(0, 0).normalized_and_length()
        assert normalized == Vec2d(0, 0)
        assert length == 0

        with pytest.raises(AttributeError):
            v.length = 5  # type: ignore

        v2 = Vec2d(10, -2)
        assert v.distance(v2) == (v - v2).length

    def testAnglesDegrees(self) -> None:
        v = Vec2d(0, 3)
        assert v.angle == 90

        v2 = Vec2d(*v)
        v = v.rotated(-90)
        assert v.angle_between(v2) == 90

        v2 = v2.rotated(-90)
        assert v.length == v2.length
        assert v2.angle == approx(0)
        assert v2.x == approx(3)
        assert v2.y == approx(0)
        assert (v - v2).length < 0.00001
        assert v.length == v2.length

        v2 = v2.rotated(300)
        assert v.angle_between(v2) == approx(-60)

        v2 = v2.rotated(v2.angle_between(v))
        assert v.angle_between(v2) == approx(0)

    def testAnglesRadians(self) -> None:
        v = Vec2d(0, 3)
        assert v.angle_radians == approx(math.pi / 2.0)

        v2 = Vec2d(*v)
        v = v.rotated(math.degrees(-math.pi / 2.0))
        assert v.angle_radians_between(v2) == approx(math.pi / 2.0)

        v2 = v2.rotated_radians(-math.pi / 2.0)
        assert v.length == approx(v2.length)
        assert v2.angle == approx(0, abs=1e-6)
        assert v2.x == 3
        assert v2.y == approx(0)
        assert (v - v2).length < 0.00001
        assert v.length == approx(v2.length)

        v2 = v2.rotated_radians(math.pi / 3.0 * 5.0)
        assert v.angle_radians_between(v2) == approx(-math.pi / 3.0)

        v2 = v2.rotated_radians(v2.angle_radians_between(v))
        assert v.angle_radians_between(v2) == approx(0)

    def testHighLevel(self) -> None:
        basis0 = Vec2d(5.0, 0)
        basis1 = Vec2d(0, 0.5)
        v = Vec2d(10, 1)
        assert v.convert_to_basis(basis0, basis1) == (2, 2)
        assert v.projection(basis0) == (10, 0)
        assert v.projection(Vec2d(0, 0)) == (0, 0)
        assert v.projection((0, 0)) == (0, 0)
        assert basis0.dot(basis1) == 0

    def testCross(self) -> None:
        lhs = Vec2d(1, 0.5)
        rhs = Vec2d(4, 6)
        assert lhs.cross(rhs) == 4

    def testComparison(self) -> None:
        int_vec = Vec2d(3, -2)
        flt_vec = Vec2d(3.0, -2.0)
        zero_vec = Vec2d(0, 0)
        assert int_vec == flt_vec
        assert int_vec != zero_vec
        assert not (flt_vec == zero_vec)
        assert not (flt_vec != int_vec)
        assert int_vec == (3, -2)
        assert int_vec != (0, 0)
        assert int_vec != 5  # type: ignore
        assert int_vec != (3, -2, -5)  # type: ignore

    def testImmutable(self) -> None:
        inplace_vec = Vec2d(5, 13)
        inplace_ref = inplace_vec
        inplace_vec *= 0.5
        inplace_vec += Vec2d(0.5, 0.5)
        inplace_vec -= Vec2d(3.5, 3.5)
        inplace_vec /= 5
        assert inplace_ref == Vec2d(5, 13)
        assert inplace_vec == Vec2d(-0.1, 0.7)

    def testPickle(self) -> None:
        testvec = Vec2d(5, 0.3)
        testvec_str = pickle.dumps(testvec)
        loaded_vec = pickle.loads(testvec_str)
        assert testvec == loaded_vec
