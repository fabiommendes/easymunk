import pickle
import unittest
from typing import Any
from pytest import approx
import easymunk as mk


class UnitTestShape(unittest.TestCase):
    def testId(self) -> None:
        c = mk.Circle(None, 4)
        self.assertGreater(c._id, 0)

    def testPointQuery(self) -> None:
        b = mk.Body(1, 1)
        c = mk.Circle(b, 5)
        c.cache_bb()

        info = c.point_query((0, 0))

        assert info.shape == c
        assert info.point == (0, 0)
        assert info.distance == -5
        assert info.gradient == (0, 1)

        info = c.point_query((11, 0))

        assert info.shape == c
        assert info.point == (5, 0)
        assert info.distance == 6
        assert info.gradient == (1, 0)

    def testSegmentQuery(self) -> None:
        b = mk.Body(1, 1, position=(10, 10))
        c = mk.Circle(b, 5)
        c.cache_bb()
        assert c.segment_query((-50, 0), (50, 0)) is None

        info = c.segment_query((-50, 0), (50, 0), 5)
        assert info.shape == c
        assert info.point == (10, 5)
        assert info.normal == (0, -1)
        assert info.alpha == 0.6

        info = c.segment_query((-50, -50), (50, 50), 0)
        assert info.point == approx((6.464, 6.464), abs=1e-2)
        assert info.normal == approx((-0.707, -0.707), abs=1e-2)
        assert info.alpha == approx(0.5646446609406)

    def testMass(self) -> None:
        c = mk.Circle(None, 1)
        assert c.mass == 0
        c.mass = 2
        assert c.mass == 2

    def testDensity(self) -> None:
        c = mk.Circle(None, 1)
        assert c.density == 0
        c.density = 2
        assert c.density == 2

    def testMoment(self) -> None:
        c = mk.Circle(None, 5)
        assert c.moment == 0

        c.density = 2
        assert c.moment == approx(1963.4954084936207)

        c.density = 0
        c.mass = 2
        assert c.moment == approx(25)

    def testArea(self) -> None:
        c = mk.Circle(None, 5)
        assert c.area == 78.53981633974483

    def testCenterOfGravity(self) -> None:
        c = mk.Circle(None, 5)
        assert c.center_of_gravity == (0, 0)

        c = mk.Circle(None, 5, (10, 5))
        assert c.center_of_gravity.x == 10
        assert c.center_of_gravity.y == 5

    def testNoBody(self) -> None:
        c = mk.Circle(None, 1)
        assert c.body is None

    def testRemoveBody(self) -> None:
        b = mk.Body(1, 1)
        c = mk.Circle(b, 1)
        c.body = None

        assert c.body is None
        assert len(b.shapes) == 0

    def testSwitchBody(self) -> None:
        b1 = mk.Body(1, 1)
        b2 = mk.Body(1, 1)
        c = mk.Circle(b1, 1)
        assert c.body == b1
        assert c in b1.shapes
        assert c not in b2.shapes
        c.body = b2
        assert c.body == b2
        assert c not in b1.shapes
        assert c in b2.shapes

    def testSensor(self) -> None:
        b1 = mk.Body(1, 1)
        c = mk.Circle(b1, 1)
        assert not c.sensor

        c.sensor = True
        assert c.sensor

    def testElasticity(self) -> None:
        b1 = mk.Body(1, 1)
        c = mk.Circle(b1, 1)
        assert c.elasticity == 0
        c.elasticity = 1
        assert c.elasticity == 1

    def testFriction(self) -> None:
        b1 = mk.Body(1, 1)
        c = mk.Circle(b1, 1)
        assert c.friction == 0
        c.friction = 1
        assert c.friction == 1

    def testSurfaceVelocity(self) -> None:
        b1 = mk.Body(1, 1)
        c = mk.Circle(b1, 1)
        assert c.surface_velocity == (0, 0)
        c.surface_velocity = (1, 2)
        assert c.surface_velocity == (1, 2)

    def testCollisionType(self) -> None:
        b1 = mk.Body(1, 1)
        c = mk.Circle(b1, 1)
        assert c.collision_type == 0
        c.collision_type = 1
        assert c.collision_type == 1

    def testFilter(self) -> None:
        b1 = mk.Body(1, 1)
        c = mk.Circle(b1, 1)
        assert c.filter == mk.ShapeFilter(0, 0xFFFFFFFF, 0xFFFFFFFF)
        c.filter = mk.ShapeFilter(1, 0xFFFFFFF2, 0xFFFFFFF3)
        assert c.filter == mk.ShapeFilter(1, 0xFFFFFFF2, 0xFFFFFFF3)

    def testSpace(self) -> None:
        b1 = mk.Body(1, 1)
        c = mk.Circle(b1, 1)
        assert c.space is None
        s = mk.Space()
        s.add(b1, c)
        assert c.space == s

    def testShapesCollide(self) -> None:
        b1 = mk.Body(1, 1)
        s1 = mk.Circle(b1, 10)

        b2 = mk.Body(1, 1)
        b2.position = 30, 30
        s2 = mk.Circle(b2, 10)

        c = s1.shapes_collide(s2)
        assert c.normal == (1, 0)
        assert len(c.points) == 1
        point = c.points[0]
        assert point.point_a == (10, 0)
        assert point.point_b == (-10, 0)
        assert point.distance == -20

    def testPickle(self) -> None:
        b = mk.Body(1, 2)
        c = mk.Circle(b, 3, (4, 5))
        c.sensor = True
        c.collision_type = 6
        c.filter = mk.ShapeFilter()
        c.elasticity = 7
        c.friction = 8
        c.surface_velocity = (9, 10)

        s = pickle.dumps(c)
        c2 = pickle.loads(s)

        assert c.sensor == c2.sensor
        assert c.collision_type == c2.collision_type
        assert c.filter == c2.filter
        assert c.elasticity == c2.elasticity
        assert c.friction == c2.friction
        assert c.surface_velocity == c2.surface_velocity
        assert c.density == c2.density
        assert c.mass == c2.mass
        assert c.body.mass == c2.body.mass

        c = mk.Circle(None, 1)
        c.density = 3

        s = pickle.dumps(c)
        c2 = pickle.loads(s)

        assert c.mass == c2.mass
        assert c.density == c2.density

        c2 = c.copy()


class UnitTestCircle(unittest.TestCase):
    def testCircleBB(self) -> None:
        b = mk.Body(10, 10)
        c = mk.Circle(b, 5)

        c.cache_bb()

        assert c.bb == mk.BB(-5.0, -5.0, 5.0, 5.0)

    def testCircleNoBody(self) -> None:
        c = mk.Circle(None, 5)

        bb = c.update_transform(mk.Transform(1, 2, 3, 4, 5, 6))
        assert c.bb == bb
        assert c.bb == mk.BB(0, 1, 10, 11)

    def test_offset(self) -> None:
        c = mk.Circle(None, 5, offset=(1, 2))
        assert c.offset == (1, 2)

        c.offset = (3, 4)
        assert c.offset == (3, 4)

    def test_radius(self) -> None:
        c = mk.Circle(None, 5)
        assert c.radius == 5

        c.radius = 3
        assert c.radius == 3

    def testPickle(self) -> None:
        c = mk.Circle(None, 3, (4, 5))

        s = pickle.dumps(c)
        c2 = pickle.loads(s)

        assert c.radius == c2.radius
        assert c.offset == c2.offset


class UnitTestSegment(unittest.TestCase):
    def testBB(self) -> None:
        s = mk.Space()
        b = mk.Body(10, 10)
        c = mk.Segment(b, (2, 2), (2, 3), 2)

        c.cache_bb()

        assert c.bb == mk.BB(0, 0, 4.0, 5.0)

    def test_properties(self) -> None:
        c = mk.Segment(None, (2, 2), (2, 3), 4)

        assert c.a == (2, 2)
        assert c.b == (2, 3)
        assert c.normal == (1, 0)
        assert c.radius == 4

        c.endpoints = (3, 4), (5, 6)
        assert c.a == (3, 4)
        assert c.b == (5, 6)

        c.radius = 5
        assert c.radius == 5

    def testSetNeighbors(self) -> None:
        c = mk.Segment(None, (2, 2), (2, 3), 1)
        c.set_neighbors((2, 2), (2, 3))

    def testSegmentSegmentCollision(self) -> None:
        s = mk.Space()
        b1 = mk.Body(10, 10)
        c1 = mk.Segment(b1, (-1, -1), (1, 1), 1)
        b2 = mk.Body(10, 10)
        c2 = mk.Segment(b2, (1, -1), (-1, 1), 1)

        s.add(b1, b2, c1, c2)

        self.num_of_begins = 0

        def begin(arb: mk.Arbiter, space: mk.Space, data: Any) -> bool:
            self.num_of_begins += 1
            return True

        s.add_default_collision_handler().begin = begin
        s.step(0.1)

        assert 1 == self.num_of_begins

    def testPickle(self) -> None:
        c = mk.Segment(None, (1, 2), (3, 4), 5)

        s = pickle.dumps(c)
        c2 = pickle.loads(s)

        assert c.a == c2.a
        assert c.b == c2.b
        assert c.radius == c2.radius


class UnitTestPoly(unittest.TestCase):
    def testInit(self) -> None:
        c = mk.Poly(None, [(0, 0), (10, 10), (20, 0), (-10, 10)], None, 0)

        b = mk.Body(1, 2)
        c = mk.Poly(
            b, [(0, 0), (10, 10), (20, 0), (-10, 10)], mk.Transform.identity(), 6
        )

    def test_vertices(self) -> None:
        vs = [(-10, 10), (0, 0), (20, 0), (10, 10)]
        c = mk.Poly(None, vs, None, 0)

        assert c.get_vertices() == vs

        c2 = mk.Poly(None, vs, mk.Transform(1, 2, 3, 4, 5, 6), 0)

        vs2 = [(5.0, 6.0), (25.0, 26.0), (45.0, 66.0), (25.0, 46.0)]
        assert c2.get_vertices() == vs2

        vs2 = [(-3, 3), (0, 0), (3, 0)]
        c.set_vertices(vs2)
        assert c.get_vertices() == vs2

        vs3 = [(-4, 4), (0, 0), (4, 0)]
        c.set_vertices(vs3, mk.Transform.identity())
        assert c.get_vertices() == vs3

    def testBB(self) -> None:
        c = mk.Poly(None, [(2, 2), (4, 3), (3, 5)])
        bb = c.update_transform(mk.Transform.identity())
        assert bb == c.bb
        assert c.bb == mk.BB(2, 2, 4, 5)

        b = mk.Body(1, 2)
        c = mk.Poly(b, [(2, 2), (4, 3), (3, 5)])
        c.cache_bb()
        assert c.bb == mk.BB(2, 2, 4, 5)

        s = mk.Space()
        b = mk.Body(1, 2)
        c = mk.Poly(b, [(2, 2), (4, 3), (3, 5)])
        s.add(b, c)
        assert c.bb == mk.BB(2, 2, 4, 5)

    def test_radius(self) -> None:
        c = mk.Poly(None, [(2, 2), (4, 3), (3, 5)], radius=10)
        assert c.radius == 10

        c.radius = 20
        assert c.radius == 20

    def testCreateBox(self) -> None:
        c = mk.Poly.create_box(None, (4, 2), 3)
        assert c.get_vertices() == [(2, -1), (2, 1), (-2, 1), (-2, -1)]

        c = mk.Poly.create_box_bb(None, mk.BB(1, 2, 3, 4), 3)
        assert c.get_vertices() == [(3, 2), (3, 4), (1, 4), (1, 2)]

    def testPickle(self) -> None:
        c = mk.Poly(None, [(1, 2), (3, 4), (5, 6)], radius=5)

        s = pickle.dumps(c)
        c2 = pickle.loads(s)

        assert c.get_vertices() == c2.get_vertices()
        assert c.radius == c2.radius
