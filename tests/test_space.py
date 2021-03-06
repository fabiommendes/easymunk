from __future__ import with_statement

import copy
import io
import pickle
import sys
import unittest
import warnings
from typing import Any, Callable, Sequence

import pytest
from pytest import approx

import easymunk as p
from easymunk import *
from easymunk.constraints import *
from easymunk.vec2d import Vec2d


class UnitTestSpace(unittest.TestCase):
    def _setUp(self) -> None:
        self.s = p.Space()

        self.b1, self.b2 = p.Body(1, 3), p.Body(10, 100)
        self.s.add(self.b1, self.b2)
        self.b1.position = 10, 0
        self.b2.position = 20, 0

        self.s1, self.s2 = p.Circle(5, body=self.b1), p.Circle(10, body=self.b2)
        self.s.add(self.s1, self.s2)
        pass

    def _tearDown(self) -> None:
        del self.s
        del self.b1, self.b2
        del self.s1, self.s2
        pass

    def testProperties(self) -> None:
        s = p.Space()

        assert s.iterations == 10
        s.iterations = 15
        assert s.iterations == 15

        assert s.gravity == (0, 0)
        s.gravity = Vec2d(10, 2)
        assert s.gravity == (10, 2)
        assert s.gravity.x == 10

        assert s.damping == 1
        s.damping = 3
        assert s.damping == 3

        assert s.idle_speed_threshold == 0
        s.idle_speed_threshold = 4
        assert s.idle_speed_threshold == 4

        assert str(s.sleep_time_threshold) == "inf"
        s.sleep_time_threshold = 5
        assert s.sleep_time_threshold == 5

        assert s.collision_slop == approx(0.1)
        s.collision_slop = 6
        assert s.collision_slop == 6

        assert s.collision_bias == approx(0.0017970074436)
        s.collision_bias = 0.2
        assert s.collision_bias == 0.2

        assert s.collision_persistence == 3
        s.collision_persistence = 9
        assert s.collision_persistence == 9

        assert s.current_time_step == 0
        s.step(0.1)
        assert s.current_time_step == 0.1

        assert s.static_body is not None
        assert s.static_body.body_type == p.Body.STATIC

        assert s.threads == 1
        s.threads = 2
        assert s.threads == 1

    def testThreaded(self) -> None:
        s = p.Space(threaded=True)
        s.step(1)
        s.threads = 2
        import platform

        if platform.system() == "Windows":
            assert s.threads == 1
        else:
            assert s.threads == 2
        s.step(1)

    def testSpatialHash(self) -> None:
        s = p.Space()
        s.use_spatial_hash(10, 100)
        s.step(1)
        s.add(p.Body(1, 2))
        s.step(1)

    def testAddRemove(self) -> None:
        s = p.Space()
        assert s.bodies == []
        assert s.shapes == []

        b = p.Body(1, 2)
        s.add(b)
        assert s.bodies == [b]
        assert s.shapes == []

        c1 = p.Circle(10, body=b)
        s.add(c1)
        assert s.bodies == [b]
        assert s.shapes == [c1]

        c2 = p.Circle(15, body=b)
        s.add(c2)
        assert len(s.shapes) == 2
        assert c1 in s.shapes
        assert c2 in s.shapes

        s.remove(c1)
        assert s.shapes == [c2]

        s.remove(c2, b)
        assert s.bodies == []
        assert s.shapes == []

    def testAddRemoveFromBody(self) -> None:
        s = p.Space()
        b = p.Body(1, 2)
        c1 = p.Circle(5, body=b)
        c2 = p.Circle(10, body=b)
        s.add(b)
        assert s.bodies == [b]
        assert s.shapes == {c1, c2}

    def testAddRemoveInStep(self) -> None:
        s = p.Space()

        b1 = p.Body(1, 2)
        c1 = p.Circle(2, body=b1)

        b2 = p.Body(1, 2)
        c2 = p.Circle(2, body=b2)

        s.add(b1, b2, c1, c2)

        b = p.Body(1, 2)
        c = p.Circle(2, body=b)

        def pre_solve_add(arb: p.Arbiter, space: p.Space, data: Any) -> bool:
            space.add(b, c)
            space.add(c, b)
            assert b not in s.bodies
            assert c not in s.shapes
            return True

        def pre_solve_remove(arb: p.Arbiter, space: p.Space, data: Any) -> bool:
            space.remove(b, c)
            space.remove(c, b)
            assert b in s.bodies
            assert c in s.shapes
            return True

        s.collision_handler(0, 0).pre_solve = pre_solve_add

        s.step(0.1)
        assert b in s.bodies
        assert c in s.shapes

        s.collision_handler(0, 0).pre_solve = pre_solve_remove

        s.step(0.1)

        assert b not in s.bodies
        assert c not in s.shapes

    def testRemoveInStep(self) -> None:
        self._setUp()
        s = self.s

        def pre_solve(arb: p.Arbiter, space: p.Space, data: Any) -> bool:
            space.remove(*arb.shapes)
            return True

        s.collision_handler(0, 0).pre_solve = pre_solve

        s.step(0.1)

        assert self.s1 not in s.shapes
        assert self.s2 not in s.shapes
        self._tearDown()

    def testPointQueryNearestWithShapeFilter(self) -> None:
        s = p.Space()
        b1 = p.Body(1, 1)
        s1 = p.Circle(10, body=b1)
        s.add(b1, s1)

        tests = [
            {"c1": 0b00, "m1": 0b00, "c2": 0b00, "m2": 0b00, "hit": 0},
            {"c1": 0b01, "m1": 0b01, "c2": 0b01, "m2": 0b01, "hit": 1},
            {"c1": 0b10, "m1": 0b01, "c2": 0b01, "m2": 0b10, "hit": 1},
            {"c1": 0b01, "m1": 0b01, "c2": 0b11, "m2": 0b11, "hit": 1},
            {"c1": 0b11, "m1": 0b00, "c2": 0b11, "m2": 0b00, "hit": 0},
            {"c1": 0b00, "m1": 0b11, "c2": 0b00, "m2": 0b11, "hit": 0},
            {"c1": 0b01, "m1": 0b10, "c2": 0b10, "m2": 0b00, "hit": 0},
            {"c1": 0b01, "m1": 0b10, "c2": 0b10, "m2": 0b10, "hit": 0},
            {"c1": 0b01, "m1": 0b10, "c2": 0b10, "m2": 0b01, "hit": 1},
            {"c1": 0b01, "m1": 0b11, "c2": 0b00, "m2": 0b10, "hit": 0},
        ]

        for test in tests:
            f1 = p.ShapeFilter(categories=test["c1"], mask=test["m1"])
            f2 = p.ShapeFilter(categories=test["c2"], mask=test["m2"])
            s1.filter = f1
            hit = s.point_query_nearest((0, 0), 0, f2)
            self.assertEqual(
                hit is not None,
                test["hit"],
                "Got {}!=None, expected {} for test: {}".format(hit, test["hit"], test),
            )

    def testPointQuery(self) -> None:
        s = p.Space()
        b1 = p.Body(1, 1)
        b1.position = 19, 0
        s1 = p.Circle(10, body=b1)
        s.add(b1, s1)

        b2 = p.Body(1, 1)
        b2.position = 0, 0
        s2 = p.Circle(10, body=b2)
        s.add(b2, s2)
        s1.filter = p.ShapeFilter(categories=0b10, mask=0b01)
        hits = s.point_query((23, 0), 0, p.ShapeFilter(categories=0b01, mask=0b10))

        assert len(hits) == 1
        assert hits[0].shape == s1
        assert hits[0].point, 29 == 0
        assert hits[0].distance == -6
        assert hits[0].gradient, 1 == 0

        hits = s.point_query((30, 0), 0, p.ShapeFilter())
        assert len(hits) == 0

        hits = s.point_query((30, 0), 30, p.ShapeFilter())
        assert len(hits) == 2
        assert hits[0].shape == s2
        assert hits[0].point == (10, 0)
        assert hits[0].distance == 20
        assert hits[0].gradient == (1, 0)

        assert hits[1].shape == s1
        assert hits[1].point == (29, 0)
        assert hits[1].distance == 1
        assert hits[1].gradient == (1, 0)

    def testPointQuerySensor(self) -> None:
        s = p.Space()
        c = p.Circle(10, body=s.static_body)
        c.sensor = True
        s.add(c)
        hits = s.point_query((0, 0), 100, p.ShapeFilter())
        assert len(hits) == 1

    def testPointQueryNearest(self) -> None:
        s = p.Space()
        b1 = p.Body(1, 1)
        b1.position = 19, 0
        s1 = p.Circle(10, body=b1)
        s.add(b1, s1)

        hit = s.point_query_nearest((23, 0), 0, p.ShapeFilter())
        assert hit is not None
        assert hit.shape == s1
        assert hit.point == (29, 0)
        assert hit.distance == -6
        assert hit.gradient == (1, 0)

        hit = s.point_query_nearest((30, 0), 0, p.ShapeFilter())
        assert hit is None

        hit = s.point_query_nearest((30, 0), 10, p.ShapeFilter())
        assert hit is not None
        assert hit.shape == s1
        assert hit.point == (29, 0)
        assert hit.distance == 1
        assert hit.gradient == (1, 0)

    def testPointQueryNearestSensor(self) -> None:
        s = p.Space()
        c = p.Circle(10, body=s.static_body)
        c.sensor = True
        s.add(c)
        hit = s.point_query_nearest((0, 0), 100, p.ShapeFilter())
        assert hit is None

    def testBBQuery(self) -> None:
        s = p.Space()

        b1 = p.Body(1, 1)
        b1.position = 19, 0
        s1 = p.Circle(10, body=b1)
        s.add(b1, s1)

        b2 = p.Body(1, 1)
        b2.position = 0, 0
        s2 = p.Circle(10, body=b2)
        s.add(b2, s2)

        bb = p.BB(-7, -7, 7, 7)
        hits = s.bb_query(bb, p.ShapeFilter())
        assert len(hits) == 1
        assert s2 in hits
        assert s1 not in hits

    def testBBQuerySensor(self) -> None:
        s = p.Space()
        c = p.Circle(10, body=s.static_body)
        c.sensor = True
        s.add(c)
        hits = s.bb_query(p.BB(0, 0, 10, 10), p.ShapeFilter())
        assert len(hits) == 1

    def testShapeQuery(self) -> None:
        self._setUp()
        b = p.Body(body_type=p.Body.KINEMATIC)
        s = p.Circle(2, body=b)
        b.position = 20, 1

        hits = self.s.shape_query(s)

        assert len(hits) == 1
        assert self.s2 == hits[0].shape
        self._tearDown()

    def testShapeQuerySensor(self) -> None:
        s = p.Space()
        c = p.Circle(10, body=s.static_body)
        c.sensor = True
        s.add(c)
        hits = s.shape_query(p.Circle(200, body=None))
        assert len(hits) == 1

    def testStaticPointQueries(self) -> None:
        self._setUp()
        b = p.Body(body_type=p.Body.KINEMATIC)
        c = p.Circle(10, body=b)
        b.position = -50, -50

        self.s.add(b, c)

        hit = self.s.point_query_nearest((-50, -55), 0, p.ShapeFilter())
        assert hit is not None
        assert hit.shape == c

        hits = self.s.point_query((-50, -55), 0, p.ShapeFilter())
        assert hits[0].shape == c
        self._tearDown()

    def testReindexShape(self) -> None:
        s = p.Space()

        b = p.Body(body_type=p.Body.KINEMATIC)
        c = p.Circle(10, body=b)

        s.add(b, c)

        b.position = -50, -50
        hit = s.point_query_nearest((-50, -55), 0, p.ShapeFilter())
        assert hit is None
        s.reindex_shape(c)
        hit = s.point_query_nearest((-50, -55), 0, p.ShapeFilter())
        assert hit is not None
        assert hit.shape == c

    def testReindexShapesForBody(self) -> None:
        s = p.Space()
        b = p.Body(body_type=p.Body.STATIC)
        c = p.Circle(10, body=b)

        s.add(b, c)

        b.position = -50, -50
        hit = s.point_query_nearest((-50, -55), 0, p.ShapeFilter())
        assert hit is None
        s.reindex_shapes_for_body(b)

        hit = s.point_query_nearest((-50, -55), 0, p.ShapeFilter())
        assert hit is not None
        assert hit.shape == c

    def testReindexStatic(self) -> None:
        s = p.Space()
        b = p.Body(body_type=p.Body.STATIC)
        c = p.Circle(10, body=b)

        s.add(b, c)

        b.position = -50, -50
        hit = s.point_query_nearest((-50, -55), 0, p.ShapeFilter())
        assert hit is None
        s.reindex_static()
        hit = s.point_query_nearest((-50, -55), 0, p.ShapeFilter())
        assert hit is not None
        assert hit.shape == c

    def testReindexStaticCollision(self) -> None:
        s = p.Space()
        b1 = p.Body(10, 1000)
        c1 = p.Circle(10, body=b1)
        b1.position = Vec2d(20, 20)

        b2 = p.Body(body_type=p.Body.STATIC)
        s2 = p.Segment((-10, 0), (10, 0), 1, b2)

        s.add(b1, c1)
        s.add(b2, s2)

        s2.endpoints = (-10, 0), (100, 0)
        s.gravity = 0, -100

        for _ in range(10):
            s.step(0.1)

        assert b1.position.y < 0

        b1.position = Vec2d(20, 20)
        b1.velocity = 0, 0
        s.reindex_static()

        for _ in range(10):
            s.step(0.1)

        assert b1.position.y > 10

    def testSegmentQuery(self) -> None:
        s = p.Space()

        b1 = p.Body(1, 1)
        b1.position = 19, 0
        s1 = p.Circle(10, body=b1)
        s.add(b1, s1)

        b2 = p.Body(1, 1)
        b2.position = 0, 0
        s2 = p.Circle(10, body=b2)
        s.add(b2, s2)

        hits = s.segment_query((-13, 0), (131, 0), 0, p.ShapeFilter())

        assert len(hits) == 2
        assert hits[0].shape == s2
        assert hits[0].point == (-10, 0)
        assert hits[0].normal == (-1, 0)
        assert hits[0].alpha == approx(0.0208333333333)

        assert hits[1].shape == s1
        assert hits[1].point == (9, 0)
        assert hits[1].normal == (-1, 0)
        assert hits[1].alpha == approx(0.1527777777777)

        hits = s.segment_query((-13, 50), (131, 50), 0, p.ShapeFilter())
        assert len(hits) == 0

    def testSegmentQuerySensor(self) -> None:
        s = p.Space()
        c = p.Circle(10, body=s.static_body)
        c.sensor = True
        s.add(c)
        hits = s.segment_query((-20, 0), (20, 0), 1, p.ShapeFilter())
        assert len(hits) == 1

    def testSegmentQueryFirst(self) -> None:
        s = p.Space()

        b1 = p.Body(1, 1)
        b1.position = 19, 0
        s1 = p.Circle(10, body=b1)
        s.add(b1, s1)

        b2 = p.Body(1, 1)
        b2.position = 0, 0
        s2 = p.Circle(10, body=b2)
        s.add(b2, s2)

        hit = s.segment_query_first((-13, 0), (131, 0), 0, p.ShapeFilter())

        assert hit is not None
        assert hit.shape == s2
        assert hit.point == (-10, 0)
        assert hit.normal == (-1, 0)
        assert hit.alpha == approx(0.0208333333333)

        hit = s.segment_query_first((-13, 50), (131, 50), 0, p.ShapeFilter())
        assert hit is None

    def testSegmentQueryFirstSensor(self) -> None:
        s = p.Space()
        c = p.Circle(10, body=s.static_body)
        c.sensor = True
        s.add(c)
        hit = s.segment_query_first((-20, 0), (20, 0), 1, p.ShapeFilter())
        self.assertIsNone(hit)

    def testStaticSegmentQueries(self) -> None:
        self._setUp()
        b = p.Body(body_type=p.Body.KINEMATIC)
        c = p.Circle(10, body=b)
        b.position = -50, -50

        self.s.add(b, c)

        hit = self.s.segment_query_first((-70, -50), (-30, -50), 0, p.ShapeFilter())
        assert hit is not None
        assert hit.shape == c
        hits = self.s.segment_query((-70, -50), (-30, -50), 0, p.ShapeFilter())
        assert hits[0].shape == c
        self._tearDown()

    def testCollisionHandlerBegin(self) -> None:
        s = p.Space()
        b1 = p.Body(1, 1)
        c1 = p.Circle(10, body=b1)
        b2 = p.Body(1, 1)
        c2 = p.Circle(10, body=b2)
        s.add(b1, c1, b2, c2)

        self.hits = 0

        def begin(arb: p.Arbiter, space: p.Space, data: Any) -> bool:
            self.hits += h.data["test"]
            return True

        h = s.collision_handler(0, 0)
        h.data["test"] = 1
        h.begin = begin

        for x in range(10):
            s.step(0.1)

        assert self.hits == 1

    def testCollisionHandlerBeginNoReturn(self) -> None:
        s = p.Space()
        b1 = p.Body(1, 1)
        c1 = p.Circle(10, body=b1)
        b2 = p.Body(1, 1)
        c2 = p.Circle(10, body=b2)
        s.add(b1, c1, b2, c2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            def begin(arb: p.Arbiter, space: p.Space, data: Any) -> bool:
                return  # type: ignore

            s.collision_handler(0, 0).begin = begin
            s.step(0.1)

            assert w is not None
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)

    def testCollisionHandlerPreSolve(self) -> None:
        s = p.Space()
        b1 = p.Body(1, 1)
        c1 = p.Circle(10, body=b1)
        c1.collision_type = 1
        b2 = p.Body(1, 1)
        c2 = p.Circle(10, body=b2)
        s.add(b1, c1, b2, c2)

        d = {}

        def pre_solve(arb: p.Arbiter, space: p.Space, data: Any) -> bool:
            d["shapes"] = arb.shapes
            d["space"] = space  # type: ignore
            d["test"] = data["test"]
            return True

        h = s.collision_handler(0, 1)
        h.data["test"] = 1
        h.pre_solve = pre_solve
        s.step(0.1)
        assert c1 == d["shapes"][1]
        assert c2 == d["shapes"][0]
        assert s == d["space"]
        assert 1 == d["test"]

    def testCollisionHandlerPreSolveNoReturn(self) -> None:
        s = p.Space()
        b1 = p.Body(1, 1)
        c1 = p.Circle(10, body=b1)
        b2 = p.Body(1, 1)
        c2 = p.Circle(10, body=b2)
        s.add(b1, c1, b2, c2)

        def pre_solve(arb: p.Arbiter, space: p.Space, data: Any) -> bool:
            return  # type: ignore

        s.collision_handler(0, 0).pre_solve = pre_solve

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            s.step(0.1)
            assert w is not None
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)

    def testCollisionHandlerPostSolve(self) -> None:
        self._setUp()
        self.hit = 0

        def post_solve(arb: p.Arbiter, space: p.Space, data: Any) -> None:
            self.hit += 1

        self.s.collision_handler(0, 0).post_solve = post_solve
        self.s.step(0.1)
        assert self.hit == 1
        self._tearDown()

    def testCollisionHandlerSeparate(self) -> None:
        s = p.Space()

        b1 = p.Body(1, 1)
        c1 = p.Circle(10, body=b1)
        b1.position = 9, 11

        b2 = p.Body(body_type=p.Body.STATIC)
        c2 = p.Circle(10, body=b2)
        b2.position = 0, 0

        s.add(b1, c1, b2, c2)
        s.gravity = 0, -100

        separated = False

        def separate(arb: p.Arbiter, space: p.Space, data: Any) -> None:
            nonlocal separated
            separated = data["test"]

        h = s.collision_handler(0, 0)
        h.data["test"] = True
        h.separate = separate

        for x in range(10):
            s.step(0.1)

        assert separated

    def testCollisionHandlerRemoveSeparateAdd(self) -> None:
        s = p.Space()
        b1 = p.Body(1, 10)
        c1 = p.Circle(10, body=b1)
        c2 = p.Circle(5, body=s.static_body)

        s.add(b1, c1, c2)

        def separate(*_: Any) -> None:
            s.add(p.Circle(2, body=s.static_body))
            s.remove(c1)

        s.default_collision_handler().separate = separate

        s.step(1)
        s.remove(c1)

    def testCollisionHandlerKeyOrder(self) -> None:
        s = p.Space()
        h1 = s.collision_handler(1, 2)
        h2 = s.collision_handler(2, 1)

        assert h1 == h2

    def testWildcardCollisionHandler(self) -> None:
        s = p.Space()
        b1 = p.Body(1, 1)
        c1 = p.Circle(10, body=b1)
        b2 = p.Body(1, 1)
        c2 = p.Circle(10, body=b2)
        s.add(b1, c1, b2, c2)

        d = {}

        def pre_solve(arb: p.Arbiter, space: p.Space, data: Any) -> bool:
            d["shapes"] = arb.shapes
            d["space"] = space  # type: ignore
            return True

        s.wildcard_collision_handler(1).pre_solve = pre_solve
        s.step(0.1)

        assert {} == d

        c1.collision_type = 1
        s.step(0.1)

        assert c1 == d["shapes"][0]
        assert c2 == d["shapes"][1]
        assert s == d["space"]

    def testDefaultCollisionHandler(self) -> None:
        s = p.Space()
        b1 = p.Body(1, 1)
        c1 = p.Circle(10, body=b1)
        c1.collision_type = 1
        b2 = p.Body(1, 1)
        c2 = p.Circle(10, body=b2)
        c2.collision_type = 2
        s.add(b1, c1, b2, c2)

        d = {}

        def pre_solve(arb: p.Arbiter, space: p.Space, data: Any) -> bool:
            d["shapes"] = arb.shapes
            d["space"] = space  # type: ignore
            return True

        s.default_collision_handler().pre_solve = pre_solve
        s.step(0.1)

        assert {c1, c2} == set(d["shapes"])
        assert s == d["space"]

    def testPostStepCallback(self) -> None:
        s = p.Space()
        b1, b2 = p.Body(1, 3), p.Body(10, 100)
        s.add(b1, b2)
        b1.position = 10, 0
        b2.position = 20, 0
        s1, s2 = p.Circle(5, body=b1), p.Circle(10, body=b2)
        s.add(s1, s2)

        self.calls = 0

        def callback(
            space: p.Space,
            key: Any,
            shapes: Sequence[Shape],
            test_self: "UnitTestSpace",
        ) -> None:
            for shape in shapes:
                s.remove(shape)
            test_self.calls += 1

        def pre_solve(arb: p.Arbiter, space: p.Space, data: Any) -> bool:
            # note that we dont pass on the whole arbiters object, instead
            # we take only the shapes.
            space.add_post_step_callback(callback, 0, arb.shapes, test_self=self)
            return True

        ch = s.collision_handler(0, 0).pre_solve = pre_solve

        s.step(0.1)
        assert [] == s.shapes
        assert self.calls == 1

        s.step(0.1)

        assert self.calls == 1

    def testDebugDraw(self) -> None:
        s = p.Space()

        b1 = p.Body(1, 3)
        s1 = p.Circle(5, body=b1)
        s.add(b1, s1)
        s.step(1)
        o = p.SpaceDebugDrawOptions()

        if sys.version_info >= (3, 0):
            new_out = io.StringIO()
        else:
            new_out = io.BytesIO()
        sys.stdout = new_out
        try:
            s.debug_draw(o)
        finally:
            sys.stdout = sys.__stdout__

        if sys.version_info >= (3, 0):
            msg = (
                "draw_circle (Vec2d(0.0, 0.0), 0.0, 5.0, "
                "SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), "
                "SpaceDebugColor(r=52.0, g=152.0, b=219.0, a=255.0))\n"
            )
        else:
            msg = (
                "('draw_circle', (Vec2d(0.0, 0.0), 0.0, 5.0, "
                "SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), "
                "SpaceDebugColor(r=52.0, g=152.0, b=219.0, a=255.0)))\n"
            )
        assert msg == new_out.getvalue()

    @unittest.skip(
        "Different behavior on windows sometimes. Expect it to be fixed in next major "
        "python version"
    )
    def testDebugDrawZeroLengthSpring(self) -> None:
        if sys.version_info < (3, 0):
            return
        s = p.Space()

        b1 = p.Body(1, 3)
        c = DampedSpring(b1, s.static_body, (0, 0), (0, 0), 0, 10, 1)
        s.add(b1, c)

        s.step(1)
        o = p.SpaceDebugDrawOptions()

        new_out = io.StringIO()
        sys.stdout = new_out
        try:
            s.debug_draw(o)
        finally:
            sys.stdout = sys.__stdout__

        expected = (
            "draw_dot (5.0, Vec2d(0.0, 0.0), SpaceDebugColor(r=142.0, g=68.0, b=173.0, "
            "a=255.0))\n"
            "draw_dot (5.0, Vec2d(0.0, 0.0), SpaceDebugColor(r=142.0, g=68.0, b=173.0, "
            "a=255.0)) \n"
            "draw_segment (Vec2d(0.0, 0.0), Vec2d(0.0, 0.0), SpaceDebugColor(r=142.0, "
            "g=68.0, b=173.0, a=255.0))\n"
            "draw_segment (Vec2d(0.0, 0.0), Vec2d(0.0, 0.0), SpaceDebugColor(r=142.0, "
            "g=68.0, b=173.0, a=255.0))\n"
        )

        actual = new_out.getvalue()
        try:
            assert expected == actual
        except:
            print("\nExpected", expected)
            print("\nActual", actual)
            raise

    def testPickleMethods(self) -> None:
        self._testCopyMethod(lambda x: pickle.loads(pickle.dumps(x)))

    def testDeepCopyMethods(self) -> None:
        self._testCopyMethod(lambda x: copy.deepcopy(x))

    @pytest.mark.skip("still segfaulting")
    def testCopyMethods(self) -> None:
        self._testCopyMethod(lambda x: x.copy())

    def _testCopyMethod(self, copy_func: Callable[[Space], Space]) -> None:
        s = p.Space(threaded=True)
        s.iterations = 2
        s.gravity = 3, 4
        s.damping = 5
        s.idle_speed_threshold = 6
        s.sleep_time_threshold = 7
        s.collision_slop = 8
        s.collision_bias = 9
        s.collision_persistence = 10
        s.threads = 2

        b1 = p.Body(1, 2)
        b2 = p.Body(3, 4)
        b3 = p.Body(5, 6)
        c1 = p.Circle(7, body=b1)
        c2 = p.Circle(8, body=b1)
        c3 = p.Circle(9, body=b2)
        c4 = p.Circle(10, body=s.static_body)
        s.add(b1, b2, b3, c1, c2, c3, c4)
        s.static_body.custom = "x"

        j1 = PinJoint(b1, b2)
        j2 = PinJoint(s.static_body, b2)
        s.add(j1, j2)

        h = s.default_collision_handler()
        h.begin = f1

        h = s.wildcard_collision_handler(1)
        h.pre_solve = f1

        h = s.collision_handler(1, 2)
        h.post_solve = f1

        h = s.collision_handler(3, 4)
        h.separate = f1

        s2 = copy_func(s)

        # Assert properties
        assert s.threaded == s2.threaded
        assert s.iterations == s2.iterations
        assert s.gravity == s2.gravity
        assert s.damping == s2.damping
        assert s.idle_speed_threshold == s2.idle_speed_threshold
        assert s.sleep_time_threshold == s2.sleep_time_threshold
        assert s.collision_slop == s2.collision_slop
        assert s.collision_bias == s2.collision_bias
        assert s.collision_persistence == s2.collision_persistence
        assert s.threads == s2.threads

        # Assert shapes, bodies and constraints
        assert {b.mass for b in s2.bodies} == {1, 3, 5}
        assert {c.radius for c in s2.shapes} == {7, 8, 9, 10}
        assert s.static_body.custom == s2.static_body.custom
        ja = [j.a for j in s2.constraints]
        self.assertIn(s2.static_body, ja)

        # Assert collision handlers
        h2 = s2.default_collision_handler()
        self.assertIsNotNone(h2.begin)
        self.assertIsNone(h2.pre_solve)
        self.assertIsNone(h2.post_solve)
        self.assertIsNone(h2.separate)

        h2 = s2.wildcard_collision_handler(1)
        self.assertIsNone(h2.begin)
        self.assertIsNotNone(h2.pre_solve)
        self.assertIsNone(h2.post_solve)
        self.assertIsNone(h2.separate)

        h2 = s2.collision_handler(1, 2)
        self.assertIsNone(h2.begin)
        self.assertIsNone(h2.pre_solve)
        self.assertIsNotNone(h2.post_solve)
        self.assertIsNone(h2.separate)

        h2 = s2.collision_handler(3, 4)
        self.assertIsNone(h2.begin)
        self.assertIsNone(h2.pre_solve)
        self.assertIsNone(h2.post_solve)
        self.assertIsNotNone(h2.separate)


def f1(*args: Any, **kwargs: Any) -> None:
    pass
