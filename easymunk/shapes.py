from math import pi, sin

__docformat__ = "reStructuredText"

import logging
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, TypeVar, Iterable

from .mat22 import Mat22

if TYPE_CHECKING:
    from .space import Space

from .util import void, cffi_body, inner_shapes, py_space
from ._chipmunk_cffi import ffi, lib
from ._mixins import PickleMixin, _State, TypingAttrMixing, HasBBMixin
from .bb import BB
from .body import Body
from .contact_point_set import ContactPointSet
from .query_info import PointQueryInfo, SegmentQueryInfo
from .shape_filter import ShapeFilter, shape_filter_from_cffi
from .transform import Transform
from .vec2d import Vec2d, VecLike, vec2d_from_cffi

T = TypeVar("T")


class Shape(PickleMixin, TypingAttrMixing, HasBBMixin):
    """Base class for all the shapes.

    You usually dont want to create instances of this class directly but use
    one of the specialized shapes instead (:py:class:`Circle`,
    :py:class:`Poly` or :py:class:`Segment`).

    All the shapes can be copied and pickled. If you copy/pickle a shape the
    body (if any) will also be copied.
    """

    _pickle_attrs_init = [
        *PickleMixin._pickle_attrs_init,
        "body",
    ]
    _pickle_attrs_general = [
        *PickleMixin._pickle_attrs_general,
        "sensor",
        "collision_type",
        "filter",
        "elasticity",
        "friction",
        "surface_velocity",
    ]
    _pickle_attrs_skip = [
        *PickleMixin._pickle_attrs_skip,
        "mass",
        "density",
    ]
    _init_kwargs = {*_pickle_attrs_general, *_pickle_attrs_skip}
    _space = None  # Weak ref to the space holding this body (if any)
    _body = None
    _shape = None
    _id_counter = 1

    @property
    def _id(self) -> int:
        """Unique id of the Shape

        .. note::
            Experimental API. Likely to change in future major, minor or point
            releases.
        """
        return int(ffi.cast("int", lib.cpShapeGetUserData(self._shape)))

    def _set_id(self) -> None:
        lib.cpShapeSetUserData(
            self._shape, ffi.cast("cpDataPointer", Shape._id_counter)
        )
        Shape._id_counter += 1

    mass: float = property(
        lambda self: lib.cpShapeGetMass(self._shape),
        lambda self, mass: void(lib.cpShapeSetMass(self._shape, mass)),
        doc="""The mass of this shape.

        This is useful when you let Pymunk calculate the total mass and inertia 
        of a body from the shapes attached to it. (Instead of setting the body 
        mass and inertia directly)
        """,
    )
    density: float = property(
        lambda self: lib.cpShapeGetDensity(self._shape),
        lambda self, density: void(lib.cpShapeSetDensity(self._shape, density)),
        doc="""The density of this shape.
        
        This is useful when you let Pymunk calculate the total mass and inertia 
        of a body from the shapes attached to it. (Instead of setting the body 
        mass and inertia directly)
        """,
    )
    moment: float = property(
        lambda self: lib.cpShapeGetMoment(self._shape),
        doc="The calculated moment of this shape.",
    )
    area: float = property(
        lambda self: lib.cpShapeGetArea(self._shape),
        doc="The calculated area of this shape.",
    )
    center_of_gravity: Vec2d = property(
        lambda self: vec2d_from_cffi(lib.cpShapeGetCenterOfGravity(self._shape)),
        doc="""The calculated center of gravity of this shape.""",
    )
    sensor: bool = property(
        lambda self: bool(lib.cpShapeGetSensor(self._shape)),
        lambda self, is_sensor: void(lib.cpShapeSetSensor(self._shape, is_sensor)),
        doc="""A boolean value if this shape is a sensor or not.

        Sensors only call collision callbacks, and never generate real
        collisions.
        """,
    )
    collision_type: int = property(
        lambda self: lib.cpShapeGetCollisionType(self._shape),
        lambda self, t: void(lib.cpShapeSetCollisionType(self._shape, t)),
        doc="""User defined collision type for the shape.

        See :py:meth:`Space.add_collision_handler` function for more 
        information on when to use this property.
        """,
    )
    filter: ShapeFilter = property(
        lambda self: shape_filter_from_cffi(lib.cpShapeGetFilter(self._shape)),
        lambda self, f: void(lib.cpShapeSetFilter(self._shape, f)),
        doc="Set the collision :py:class:`ShapeFilter` for this shape.",
    )
    elasticity: float = property(
        lambda self: lib.cpShapeGetElasticity(self._shape),
        lambda self, e: void(lib.cpShapeSetElasticity(self._shape, e)),
        doc="""Elasticity of the shape.

        A value of 0.0 gives no bounce, while a value of 1.0 will give a
        'perfect' bounce. However due to inaccuracies in the simulation
        using 1.0 or greater is not recommended.
        """,
    )
    friction: float = property(
        lambda self: lib.cpShapeGetFriction(self._shape),
        lambda self, u: void(lib.cpShapeSetFriction(self._shape, u)),
        doc="""Friction coefficient.

        Pymunk uses the Coulomb friction model, a value of 0.0 is
        frictionless.

        A value over 1.0 is perfectly fine.

        Some real world example values from Wikipedia (Remember that
        it is what looks good that is important, not the exact value).

        ==============  ======  ========
        Material        Other   Friction
        ==============  ======  ========
        Aluminium       Steel   0.61
        Copper          Steel   0.53
        Brass           Steel   0.51
        Cast iron       Copper  1.05
        Cast iron       Zinc    0.85
        Concrete (wet)  Rubber  0.30
        Concrete (dry)  Rubber  1.0
        Concrete        Wood    0.62
        Copper          Glass   0.68
        Glass           Glass   0.94
        Metal           Wood    0.5
        Polyethene      Steel   0.2
        Steel           Steel   0.80
        Steel           Teflon  0.04
        Teflon (PTFE)   Teflon  0.04
        Wood            Wood    0.4
        ==============  ======  ========
        """,
    )
    surface_velocity: Vec2d = property(
        lambda self: vec2d_from_cffi(lib.cpShapeGetSurfaceVelocity(self._shape)),
        lambda self, surface_v: void(
            lib.cpShapeSetSurfaceVelocity(self._shape, surface_v)
        ),
        doc="""The surface velocity of the object.

        Useful for creating conveyor belts or players that move around. This
        value is only used when calculating friction, not resolving the
        collision.
        """,
    )

    def _set_body(self, body: Optional["Body"]) -> None:
        if self._body is not None:
            inner_shapes(self._body).remove(self)

        lib.cpShapeSetBody(self._shape, cffi_body(body))
        if body is not None:
            inner_shapes(body).add(self),
        self._body = body

    body: Optional["Body"] = property(
        lambda self: self._body,
        _set_body,
        doc="""The body this shape is attached to. Can be set to None to
        indicate that this shape doesnt belong to a body.""",
    )

    # noinspection PyProtectedMember
    @property
    def bb(self) -> BB:
        """The bounding box :py:class:`BB` of the shape.

        Only guaranteed to be valid after :py:meth:`Shape.cache_bb` or
        :py:meth:`Space.step` is called. Moving a body that a shape is
        connected to does not update it's bounding box. For shapes used for
        queries that aren't attached to bodies, you can also use
        :py:meth:`Shape.update`.
        """
        _bb = lib.cpShapeGetBB(self._shape)
        return BB(_bb.l, _bb.b, _bb.r, _bb.t)

    @property
    def space(self) -> Optional["Space"]:
        """Get the :py:class:`Space` that shape has been added to (or
        None).
        """
        if self._space is not None:
            return py_space(self._space)
        else:
            return None

    def _init(
        self,
        body: Optional["Body"],
        _shape: ffi.CData,
        space: Optional["Space"] = None,
        **kwargs,
    ) -> None:
        self._body = body
        if body is not None:
            inner_shapes(body).add(self)
        self._shape = ffi.gc(_shape, cffi_free_shape)
        self._set_id()

        if not self._init_kwargs.issuperset(kwargs):
            keys = self._init_kwargs.difference(kwargs)
            raise TypeError(f"invalid parameters: {keys}")
        for k, v in kwargs.items():
            setattr(self, k, v)
        if space is not None:
            space.add(self)

    def _iter_bounding_boxes(self) -> Iterable["BB"]:
        return self.bb

    def __getstate__(self) -> _State:
        """Return the state of this object

        This method allows the usage of the :mod:`copy` and :mod:`pickle`
        modules with this class.
        """
        d = super().__getstate__()

        if self.mass > 0:
            d["general"].append(("mass", self.mass))
        if self.density > 0:
            d["general"].append(("density", self.density))

        return d

    def update_transform(self, transform: Transform) -> BB:
        """Update, cache and return the bounding box of a shape with an
        explicit transformation.

        Useful if you have a shape without a body and want to use it for
        querying.
        """
        _bb = lib.cpShapeUpdate(self._shape, transform)
        return BB(_bb.l, _bb.b, _bb.r, _bb.t)

    def cache_bb(self) -> BB:
        """Update and returns the bounding box of this shape"""
        _bb = lib.cpShapeCacheBB(self._shape)
        return BB(_bb.l, _bb.b, _bb.r, _bb.t)

    def point_query(self, p: Tuple[float, float]) -> PointQueryInfo:
        """Check if the given point lies within the shape.

        A negative distance means the point is within the shape.

        :return: Tuple of (distance, info)
        :rtype: (float, :py:class:`PointQueryInfo`)
        """
        info = ffi.new("cpPointQueryInfo *")
        _ = lib.cpShapePointQuery(self._shape, p, info)

        ud = int(ffi.cast("int", lib.cpShapeGetUserData(info.shape)))
        assert ud == self._id
        return PointQueryInfo(
            self,
            Vec2d(info.point.x, info.point.y),
            info.distance,
            Vec2d(info.gradient.x, info.gradient.y),
        )

    def segment_query(
        self, start: VecLike, end: VecLike, radius: float = 0
    ) -> SegmentQueryInfo:
        """Check if the line segment from start to end intersects the shape.

        :rtype: :py:class:`SegmentQueryInfo`
        """
        info = ffi.new("cpSegmentQueryInfo *")
        r = lib.cpShapeSegmentQuery(self._shape, start, end, radius, info)
        if r:
            ud = int(ffi.cast("int", lib.cpShapeGetUserData(info.shape)))
            assert ud == self._id
            return SegmentQueryInfo(
                self,
                Vec2d(info.point.x, info.point.y),
                Vec2d(info.normal.x, info.normal.y),
                info.alpha,
            )
        else:
            return SegmentQueryInfo(
                None,
                Vec2d(info.point.x, info.point.y),
                Vec2d(info.normal.x, info.normal.y),
                info.alpha,
            )

    def shapes_collide(self, b: "Shape") -> ContactPointSet:
        """Get contact information about this shape and shape b."""
        _points = lib.cpShapesCollide(self._shape, b._shape)
        return ContactPointSet._from_cp(_points)


class Circle(Shape):
    """A circle shape defined by a radius

    This is the fastest and simplest collision shape
    """

    _pickle_attrs_init = Shape._pickle_attrs_init + ["radius", "offset"]
    radius: float = property(
        lambda self: lib.cpCircleShapeGetRadius(self._shape),
        lambda self, r: void(lib.cpCircleShapeSetRadius(self._shape, r)),
        doc="""The Radius of the circle
        
       .. note::
            Changes in radius are only picked up as a change to the position
            of the shape's surface, but not it's velocity. Changing it will
            not result in realistic physical behavior. Only use if you know
            what you are doing!
        """,
    )
    offset: Vec2d = property(
        lambda self: vec2d_from_cffi(lib.cpCircleShapeGetOffset(self._shape)),
        lambda self, o: void(lib.cpCircleShapeSetOffset(self._shape, o)),
        doc="""Offset. (body space coordinates)
        
        .. note::
            Changes in offset are only picked up as a change to the position
            of the shape's surface, but not it's velocity. Changing it will
            not result in realistic physical behavior. Only use if you know
            what you are doing!
        """,
    )

    def __init__(
        self,
        body: Optional["Body"],
        radius: float,
        offset: VecLike = (0, 0),
        **kwargs,
    ) -> None:
        """body is the body attach the circle to, offset is the offset from the
        body's center of gravity in body local coordinates.

        It is legal to send in None as body argument to indicate that this
        shape is not attached to a body. However, you must attach it to a body
        before adding the shape to a space or used for a space shape query.
        """
        _shape = lib.cpCircleShapeNew(cffi_body(body), radius, offset)
        self._init(body, _shape, **kwargs)

    def _iter_bounding_boxes(self) -> Iterable["BB"]:
        yield self.bb


# noinspection PyShadowingBuiltins
class Segment(Shape):
    """A line segment shape between two points

    Meant mainly as a static shape. Can be beveled in order to give them a
    thickness.
    """

    _pickle_attrs_init = Shape._pickle_attrs_init + ["a", "b", "radius"]
    radius: float = property(
        lambda self: lib.cpSegmentShapeGetRadius(self._shape),
        lambda self, r: void(lib.cpSegmentShapeSetRadius(self._shape, r)),
        doc="""The radius/thickness of the segment
        
       .. note::
            Changes in radius are only picked up as a change to the position
            of the shape's surface, but not it's velocity. Changing it will
            not result in realistic physical behavior. Only use if you know
            what you are doing!
        """,
    )
    a: Vec2d = property(
        lambda self: vec2d_from_cffi(lib.cpSegmentShapeGetA(self._shape)),
        lambda self, a: void(lib.cpSegmentShapeSetEndpoints(self._shape, a, self.b)),
        doc="The first of the two endpoints for this segment",
    )
    b: Vec2d = property(
        lambda self: vec2d_from_cffi(lib.cpSegmentShapeGetB(self._shape)),
        lambda self, b: void(lib.cpSegmentShapeSetEndpoints(self._shape, self.a, b)),
        doc="The second of the two endpoints for this segment",
    )
    endpoints: Tuple[Vec2d, Vec2d] = property(
        lambda self: (self.a, self.b),
        lambda self, pts: void(lib.cpSegmentShapeSetEndpoints(self._shape, *pts)),
        doc="A tuple with (a, b) endpoints.",
    )
    normal: Vec2d = property(
        lambda self: vec2d_from_cffi(lib.cpSegmentShapeGetNormal(self._shape)),
        doc="The normal",
    )

    def __init__(
        self,
        body: Optional[Body],
        a: VecLike,
        b: VecLike,
        radius: float,
        **kwargs,
    ) -> None:
        """Create a Segment

        It is legal to send in None as body argument to indicate that this
        shape is not attached to a body. However, you must attach it to a body
        before adding the shape to a space or used for a space shape query.

        :param Body body: The body to attach the segment to
        :param a: The first endpoint of the segment
        :param b: The second endpoint of the segment
        :param float radius: The thickness of the segment
        """
        _shape = lib.cpSegmentShapeNew(cffi_body(body), a, b, radius)
        self._init(body, _shape, **kwargs)

    def set_neighbors(self: T, prev: VecLike, next: VecLike) -> T:
        """When you have a number of segment shapes that are all joined
        together, things can still collide with the "cracks" between the
        segments. By setting the neighbor segment endpoints you can tell
        Chipmunk to avoid colliding with the inner parts of the crack.
        """
        lib.cpSegmentShapeSetNeighbors(self._shape, prev, next)
        return self


class Poly(Shape):
    """A convex polygon shape

    Slowest, but most flexible collision shape.
    """

    radius: float = property(
        lambda self: lib.cpPolyShapeGetRadius(self._shape),
        lambda self, r: void(lib.cpPolyShapeSetRadius(self._shape, r)),
        doc="""The radius/thickness of polygon lines
        
       .. note::
            Changes in radius are only picked up as a change to the position
            of the shape's surface, but not it's velocity. Changing it will
            not result in realistic physical behavior. Only use if you know
            what you are doing!
        """,
    )

    @staticmethod
    def create_box(
        body: Optional["Body"],
        size: VecLike = (10, 10),
        radius: float = 0,
        **kwargs,
    ) -> "Poly":
        """Convenience function to create a box given a width and height.

        The boxes will always be centered at the center of gravity of the
        body you are attaching them to.  If you want to create an off-center
        box, you will need to use the normal constructor Poly(...).

        Adding a small radius will bevel the corners and can significantly
        reduce problems where the box gets stuck on seams in your geometry.

        :param Body body: The body to attach the poly to
        :param size: Size of the box as (width, height)
        :type size: (`float, float`)
        :param float radius: Radius of poly
        :rtype: :py:class:`Poly`
        """

        self = Poly.__new__(Poly)
        _shape = lib.cpBoxShapeNew(cffi_body(body), size[0], size[1], radius)
        self._init(body, _shape, **kwargs)

        return self

    @staticmethod
    def create_box_bb(
        body: Optional["Body"], bb: BB, radius: float = 0, **kwargs
    ) -> "Poly":
        """Convenience function to create a box shape from a :py:class:`BB`.

        The boxes will always be centered at the center of gravity of the
        body you are attaching them to.  If you want to create an off-center
        box, you will need to use the normal constructor Poly(..).

        Adding a small radius will bevel the corners and can significantly
        reduce problems where the box gets stuck on seams in your geometry.

        :param Body body: The body to attach the poly to
        :param BB bb: Size of the box
        :param float radius: Radius of poly
        :rtype: :py:class:`Poly`
        """

        self = Poly.__new__(Poly)
        _shape = lib.cpBoxShapeNew2(cffi_body(body), bb, radius)
        self._init(body, _shape, **kwargs)

        return self

    @staticmethod
    def create_regular_poly(
        body: Optional["Body"],
        n: int,
        size: float,
        radius: float = 0,
        angle: float = 0,
        **kwargs,
    ) -> "Poly":
        """Convenience function to create a regular polygon of n sides of a
        given size.

        The polygon will always be centered at the center of gravity of the
        body you are attaching it to. If you want to create an off-center
        box, you will need to use the normal constructor Poly(..).

        The first vertex is in the direction of the x-axis. This can be changed
        by setting a different initial angle.

        Adding a small radius will bevel the corners and can significantly
        reduce problems where the box gets stuck on seams in your geometry.

        :param Body body: The body to attach the poly to
        :param int n: Number of sides
        :param float size: Length of each side
        :param float radius: Radius of poly
        :param float angle: Rotation angle.
        :rtype: :py:class:`Poly`
        """

        inner_angle = 2 * pi / n
        distance = size / 2 / sin(inner_angle / 2)
        vertices = [Vec2d(distance, 0).rotated(angle)]
        while len(vertices) < n:
            vertices.append(vertices[-1].rotated(inner_angle))

        return Poly(body, vertices, radius=radius, **kwargs)

    def __init__(
        self,
        body: Optional["Body"],
        vertices: Sequence[VecLike],
        transform: Optional[Transform] = None,
        radius: float = 0,
        **kwargs,
    ) -> None:
        """Create a polygon.

        A convex hull will be calculated from the vertexes automatically.

        Adding a small radius will bevel the corners and can significantly
        reduce problems where the poly gets stuck on seams in your geometry.

        It is legal to send in None as body argument to indicate that this
        shape is not attached to a body. However, you must attach it to a body
        before adding the shape to a space or used for a space shape query.

        .. note::
            Make sure to put the vertices around (0,0) or the shape might
            behave strange.

            Either directly place the vertices like the below example:

            >>> import easymunk
            >>> w, h = 10, 20
            >>> vs = [(-w/2,-h/2), (w/2,-h/2), (w/2,h/2), (-w/2,h/2)]
            >>> poly_good = easymunk.Poly(None, vs)
            >>> print(poly_good.center_of_gravity)
            Vec2d(0.0, 0.0)

            Or use a transform to move them:

            >>> import easymunk
            >>> width, height = 10, 20
            >>> vs = [(0, 0), (width, 0), (width, height), (0, height)]
            >>> poly_bad = easymunk.Poly(None, vs)
            >>> print(poly_bad.center_of_gravity)
            Vec2d(5.0, 10.0)
            >>> t = easymunk.Transform(tx=-width/2, ty=-height/2)
            >>> poly_good = easymunk.Poly(None, vs, transform=t)
            >>> print(poly_good.center_of_gravity)
            Vec2d(0.0, 0.0)

        :param Body body: The body to attach the poly to
        :param [(float,float)] vertices: Define a convex hull of the polygon
            with a counterclockwise winding.
        :param Transform transform: Transform will be applied to every vertex.
        :param float radius: Set the radius of the poly shape

        """
        if transform is None:
            transform = Transform.identity()

        _shape = lib.cpPolyShapeNew(
            cffi_body(body), len(vertices), vertices, transform, radius
        )
        self._init(body, _shape, **kwargs)

    def __getstate__(self) -> _State:
        """Return the state of this object

        This method allows the usage of the :mod:`copy` and :mod:`pickle`
        modules with this class.
        """
        d = super(Poly, self).__getstate__()

        d["init"].append(("vertices", self.get_vertices()))
        d["init"].append(("transform", None))
        d["init"].append(("radius", self.radius))
        return d

    def get_vertices(self, *, world: bool = False) -> List[VecLike]:
        """Return list of vertices in local coordinates.

        Set ``world=True`` if you need the list of vertices in world coordinates.
        """
        n = lib.cpPolyShapeGetCount(self._shape)
        vs = [vec2d_from_cffi(lib.cpPolyShapeGetVert(self._shape, i)) for i in range(n)]
        if world:
            pos = self.body.position
            rot = Mat22.rotation(self.body.angle)
            return [rot.transform_vector(v) + pos for v in vs]
        return vs

    def set_vertices(
        self: T,
        vertices: Sequence[VecLike],
        transform: Optional[Transform] = None,
        *,
        world: bool = False,
    ) -> T:
        """Set the vertices of the poly.

        .. note::
            This change is only picked up as a change to the position
            of the shape's surface, but not it's velocity. Changing it will
            not result in realistic physical behavior. Only use if you know
            what you are doing!
        """
        if world:
            pos = self.body.position
            rot = Mat22.rotation(-self.body.angle)
            vertices = [rot.transform_vector(v) - pos for v in vertices]
        if transform is None:
            lib.cpPolyShapeSetVertsRaw(self._shape, len(vertices), vertices)
            return
        lib.cpPolyShapeSetVerts(self._shape, len(vertices), vertices, transform)
        return self


def cffi_free_shape(cp_shape):
    cp_space = lib.cpShapeGetSpace(cp_shape)
    if cp_space != ffi.NULL:
        logging.debug("free %s %s", cp_space, cp_shape)
        lib.cpSpaceRemoveShape(cp_space, cp_shape)

    logging.debug("free %s", cp_shape)
    lib.cpShapeSetBody(cp_shape, ffi.NULL)
    logging.debug("free%s", cp_shape)
    lib.cpShapeFree(cp_shape)
