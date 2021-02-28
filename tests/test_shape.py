import pickle
import unittest
from typing import Any

import easymunk as p


class UnitTestShape(unittest.TestCase):
    def testId(self) -> None:
        c = p.Circle(None, 4)
        self.assertGreater(c._id, 0)

    def testPointQuery(self) -> None:
        b = p.Body(10, 10)
        c = p.Circle(b, 5)
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
        s = p.Space()
        b = p.Body(10, 10)
        c = p.Circle(b, 5)
        c.cache_bb()

        info = c.segment_query((10, -50), (10, 50))
        self.assertEqual(info.shape, None)
        self.assertEqual(info.point, (10, 50))
        self.assertEqual(info.normal, (0, 0))
        self.assertEqual(info.alpha, 1.0)

        info = c.segment_query((10, -50), (10, 50), 6)
        self.assertEqual(info.shape, c)

        info = c.segment_query((0, -50), (0, 50))
        self.assertEqual(info.shape, c)
        self.assertAlmostEqual(info.point.x, 0)
        self.assertAlmostEqual(info.point.y, -5)
        self.assertAlmostEqual(info.normal.x, 0)
        self.assertAlmostEqual(info.normal.y, -1)
        self.assertEqual(info.alpha, 0.45)

    def testMass(self) -> None:
        c = p.Circle(None, 1)
        self.assertEqual(c.mass, 0)
        c.mass = 2
        self.assertEqual(c.mass, 2)

    def testDensity(self) -> None:
        c = p.Circle(None, 1)
        self.assertEqual(c.density, 0)
        c.density = 2
        self.assertEqual(c.density, 2)

    def testMoment(self) -> None:
        c = p.Circle(None, 5)
        self.assertEqual(c.moment, 0)
        c.density = 2
        self.assertAlmostEqual(c.moment, 1963.4954084936207)
        c.density = 0
        c.mass = 2
        self.assertAlmostEqual(c.moment, 25)

    def testArea(self) -> None:
        c = p.Circle(None, 5)
        self.assertEqual(c.area, 78.53981633974483)

    def testCenterOfGravity(self) -> None:
        c = p.Circle(None, 5)
        self.assertEqual(c.center_of_gravity, (0, 0))
        c = p.Circle(None, 5, (10, 5))
        self.assertEqual(c.center_of_gravity.x, 10)
        self.assertEqual(c.center_of_gravity.y, 5)

    def testNoBody(self) -> None:
        c = p.Circle(None, 1)
        self.assertEqual(c.body, None)

    def testRemoveBody(self) -> None:
        b = p.Body(1, 1)
        c = p.Circle(b, 1)
        c.body = None

        self.assertEqual(c.body, None)
        self.assertEqual(len(b.shapes), 0)

    def testSwitchBody(self) -> None:
        b1 = p.Body(1, 1)
        b2 = p.Body(1, 1)
        c = p.Circle(b1, 1)
        self.assertEqual(c.body, b1)
        self.assertTrue(c in b1.shapes)
        self.assertTrue(c not in b2.shapes)
        c.body = b2
        self.assertEqual(c.body, b2)
        self.assertTrue(c not in b1.shapes)
        self.assertTrue(c in b2.shapes)

    def testSensor(self) -> None:
        b1 = p.Body(1, 1)
        c = p.Circle(b1, 1)
        self.assertFalse(c.sensor)
        c.sensor = True
        self.assertTrue(c.sensor)

    def testElasticity(self) -> None:
        b1 = p.Body(1, 1)
        c = p.Circle(b1, 1)
        self.assertEqual(c.elasticity, 0)
        c.elasticity = 1
        self.assertEqual(c.elasticity, 1)

    def testFriction(self) -> None:
        b1 = p.Body(1, 1)
        c = p.Circle(b1, 1)
        self.assertEqual(c.friction, 0)
        c.friction = 1
        self.assertEqual(c.friction, 1)

    def testSurfaceVelocity(self) -> None:
        b1 = p.Body(1, 1)
        c = p.Circle(b1, 1)
        self.assertEqual(c.surface_velocity, (0, 0))
        c.surface_velocity = (1, 2)
        self.assertEqual(c.surface_velocity, (1, 2))

    def testCollisionType(self) -> None:
        b1 = p.Body(1, 1)
        c = p.Circle(b1, 1)
        self.assertEqual(c.collision_type, 0)
        c.collision_type = 1
        self.assertEqual(c.collision_type, 1)

    def testFilter(self) -> None:
        b1 = p.Body(1, 1)
        c = p.Circle(b1, 1)
        self.assertEqual(c.filter, p.ShapeFilter(0, 0xFFFFFFFF, 0xFFFFFFFF))
        c.filter = p.ShapeFilter(1, 0xFFFFFFF2, 0xFFFFFFF3)
        self.assertEqual(c.filter, p.ShapeFilter(1, 0xFFFFFFF2, 0xFFFFFFF3))

    def testSpace(self) -> None:
        b1 = p.Body(1, 1)
        c = p.Circle(b1, 1)
        self.assertEqual(c.space, None)
        s = p.Space()
        s.add(b1, c)
        self.assertEqual(c.space, s)

    def testShapesCollide(self) -> None:
        b1 = p.Body(1, 1)
        s1 = p.Circle(b1, 10)

        b2 = p.Body(1, 1)
        b2.position = 30, 30
        s2 = p.Circle(b2, 10)

        c = s1.shapes_collide(s2)
        self.assertEqual(c.normal, (1, 0))
        self.assertEqual(len(c.points), 1)
        point = c.points[0]
        self.assertEqual(point.point_a, (10, 0))
        self.assertEqual(point.point_b, (-10, 0))
        self.assertEqual(point.distance, -20)

    def testPickle(self) -> None:
        b = p.Body(1, 2)
        c = p.Circle(b, 3, (4, 5))
        c.sensor = True
        c.collision_type = 6
        c.filter = p.ShapeFilter()
        c.elasticity = 7
        c.friction = 8
        c.surface_velocity = (9, 10)

        s = pickle.dumps(c)
        c2 = pickle.loads(s)

        self.assertEqual(c.sensor, c2.sensor)
        self.assertEqual(c.collision_type, c2.collision_type)
        self.assertEqual(c.filter, c2.filter)
        self.assertEqual(c.elasticity, c2.elasticity)
        self.assertEqual(c.friction, c2.friction)
        self.assertEqual(c.surface_velocity, c2.surface_velocity)
        self.assertEqual(c.density, c2.density)
        self.assertEqual(c.mass, c2.mass)
        self.assertEqual(c.body.mass, c2.body.mass)

        c = p.Circle(None, 1)
        c.density = 3

        s = pickle.dumps(c)
        c2 = pickle.loads(s)

        self.assertEqual(c.mass, c2.mass)
        self.assertEqual(c.density, c2.density)

        c2 = c.copy()


class UnitTestCircle(unittest.TestCase):
    def testCircleBB(self) -> None:
        b = p.Body(10, 10)
        c = p.Circle(b, 5)

        c.cache_bb()

        self.assertEqual(c.bb, p.BB(-5.0, -5.0, 5.0, 5.0))

    def testCircleNoBody(self) -> None:
        c = p.Circle(None, 5)

        bb = c.update_transform(p.Transform(1, 2, 3, 4, 5, 6))
        self.assertEqual(c.bb, bb)
        self.assertEqual(c.bb, p.BB(0, 1, 10, 11))

    def test_offset(self) -> None:
        c = p.Circle(None, 5, offset=(1, 2))
        assert c.offset == (1, 2)

        c.offset = (3, 4)
        assert c.offset == (3, 4)

    def test_radius(self) -> None:
        c = p.Circle(None, 5)
        assert c.radius == 5

        c.radius = 3
        assert c.radius == 3

    def testPickle(self) -> None:
        c = p.Circle(None, 3, (4, 5))

        s = pickle.dumps(c)
        c2 = pickle.loads(s)

        self.assertEqual(c.radius, c2.radius)
        self.assertEqual(c.offset, c2.offset)


class UnitTestSegment(unittest.TestCase):
    def testBB(self) -> None:
        s = p.Space()
        b = p.Body(10, 10)
        c = p.Segment(b, (2, 2), (2, 3), 2)

        c.cache_bb()

        self.assertEqual(c.bb, p.BB(0, 0, 4.0, 5.0))

    def test_properties(self) -> None:
        c = p.Segment(None, (2, 2), (2, 3), 4)

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
        c = p.Segment(None, (2, 2), (2, 3), 1)
        c.set_neighbors((2, 2), (2, 3))

    def testSegmentSegmentCollision(self) -> None:
        s = p.Space()
        b1 = p.Body(10, 10)
        c1 = p.Segment(b1, (-1, -1), (1, 1), 1)
        b2 = p.Body(10, 10)
        c2 = p.Segment(b2, (1, -1), (-1, 1), 1)

        s.add(b1, b2, c1, c2)

        self.num_of_begins = 0

        def begin(arb: p.Arbiter, space: p.Space, data: Any) -> bool:
            self.num_of_begins += 1
            return True

        s.add_default_collision_handler().begin = begin
        s.step(0.1)

        self.assertEqual(1, self.num_of_begins)

    def testPickle(self) -> None:
        c = p.Segment(None, (1, 2), (3, 4), 5)

        s = pickle.dumps(c)
        c2 = pickle.loads(s)

        self.assertEqual(c.a, c2.a)
        self.assertEqual(c.b, c2.b)
        self.assertEqual(c.radius, c2.radius)


class UnitTestPoly(unittest.TestCase):
    def testInit(self) -> None:
        c = p.Poly(None, [(0, 0), (10, 10), (20, 0), (-10, 10)], None, 0)

        b = p.Body(1, 2)
        c = p.Poly(b, [(0, 0), (10, 10), (20, 0), (-10, 10)], p.Transform.identity(), 6)

    def test_vertices(self) -> None:
        vs = [(-10, 10), (0, 0), (20, 0), (10, 10)]
        c = p.Poly(None, vs, None, 0)

        assert c.get_vertices() == vs

        c2 = p.Poly(None, vs, p.Transform(1, 2, 3, 4, 5, 6), 0)

        vs2 = [(5.0, 6.0), (25.0, 26.0), (45.0, 66.0), (25.0, 46.0)]
        assert c2.get_vertices() == vs2

        vs2 = [(-3, 3), (0, 0), (3, 0)]
        c.set_vertices(vs2)
        assert c.get_vertices() == vs2

        vs3 = [(-4, 4), (0, 0), (4, 0)]
        c.set_vertices(vs3, p.Transform.identity())
        assert c.get_vertices() == vs3

    def testBB(self) -> None:
        c = p.Poly(None, [(2, 2), (4, 3), (3, 5)])
        bb = c.update_transform(p.Transform.identity())
        self.assertEqual(bb, c.bb)
        self.assertEqual(c.bb, p.BB(2, 2, 4, 5))

        b = p.Body(1, 2)
        c = p.Poly(b, [(2, 2), (4, 3), (3, 5)])
        c.cache_bb()
        self.assertEqual(c.bb, p.BB(2, 2, 4, 5))

        s = p.Space()
        b = p.Body(1, 2)
        c = p.Poly(b, [(2, 2), (4, 3), (3, 5)])
        s.add(b, c)
        self.assertEqual(c.bb, p.BB(2, 2, 4, 5))

    def test_radius(self) -> None:
        c = p.Poly(None, [(2, 2), (4, 3), (3, 5)], radius=10)
        assert c.radius == 10

        c.radius = 20
        assert c.radius == 20

    def testCreateBox(self) -> None:
        c = p.Poly.create_box(None, (4, 2), 3)
        self.assertEqual(c.get_vertices(), [(2, -1), (2, 1), (-2, 1), (-2, -1)])

        c = p.Poly.create_box_bb(None, p.BB(1, 2, 3, 4), 3)
        self.assertEqual(c.get_vertices(), [(3, 2), (3, 4), (1, 4), (1, 2)])

    def testPickle(self) -> None:
        c = p.Poly(None, [(1, 2), (3, 4), (5, 6)], radius=5)

        s = pickle.dumps(c)
        c2 = pickle.loads(s)

        self.assertEqual(c.get_vertices(), c2.get_vertices())
        self.assertEqual(c.radius, c2.radius)
