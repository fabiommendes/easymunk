__docformat__ = "reStructuredText"

import logging
from math import degrees, radians
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Set,
    TypeVar,
    Union,
    Tuple,
    List,
    Iterator,
)
from weakref import WeakSet

import sidekick.api as sk

from ._chipmunk_cffi import ffi, lib
from ._mixins import PickleMixin, HasBBMixin
from .arbiter import Arbiter
from .collections import Shapes, Constraints
from .shapes import MakeShapeMixin
from .util import void, set_attrs, py_space, init_attributes
from .vec2d import Vec2d, VecLike, vec2d_from_cffi

if TYPE_CHECKING:
    from .shape_filter import ShapeFilter
    from .space import Space
    from .constraints import Constraint
    from .shapes import Shape, Circle, Poly, Segment
    from .bb import BB
else:
    Circle = sk.import_later(".shapes:Circle", __package__)
    Segment = sk.import_later(".shapes:Segment", __package__)
    Poly = sk.import_later(".shapes:Poly", __package__)

B = TypeVar("B", bound="Body")
BodyType = int
PositionFunc = Callable[["Body", float], None]
VelocityFunc = Callable[["Body", Vec2d, float, float], None]
BODY_TYPES = {}


class Body(MakeShapeMixin, PickleMixin, HasBBMixin):
    """A rigid body

    * Use forces to modify the rigid bodies if possible. This is likely to be
      the most stable.
    * Modifying a body's velocity shouldn't necessarily be avoided, but
      applying large changes can cause strange results in the simulation.
      Experiment freely, but be warned.
    * Don't modify a body's position every step unless you really know what
      you are doing. Otherwise you're likely to get the position/velocity badly
      out of sync.

    A Body can be copied and pickled. Sleeping bodies that are copied will be
    awake in the fresh copy. When a Body is copied any spaces, shapes or
    constraints attached to the body will not be copied.
    """

    DYNAMIC = BODY_TYPES["dynamic"] = lib.CP_BODY_TYPE_DYNAMIC
    """Dynamic bodies are the default body type.

    They react to collisions,
    are affected by forces and gravity, and have a finite amount of mass.
    These are the type of bodies that you want the physics engine to
    simulate for you. Dynamic bodies interact with all types of bodies
    and can generate collision callbacks.
    """

    KINEMATIC = BODY_TYPES["kinematic"] = lib.CP_BODY_TYPE_KINEMATIC
    """Kinematic bodies are bodies that are controlled from your code
    instead of inside the physics engine.

    They arent affected by gravity and they have an infinite amount of mass
    so they don't react to collisions or forces with other bodies. Kinematic
    bodies are controlled by setting their velocity, which will cause them
    to move. Good examples of kinematic bodies might include things like
    moving platforms. Objects that are touching or jointed to a kinematic
    body are never allowed to fall asleep.
    """

    STATIC = BODY_TYPES["static"] = lib.CP_BODY_TYPE_STATIC
    """Static bodies are bodies that never (or rarely) move.

    Using static bodies for things like terrain offers a big performance
    boost over other body types- because Chipmunk doesn't need to check for
    collisions between static objects and it never needs to update their
    collision information. Additionally, because static bodies don't
    move, Chipmunk knows it's safe to let objects that are touching or
    jointed to them fall asleep. Generally all of your level geometry
    will be attached to a static body except for things like moving
    platforms or doors. Every space provide a built-in static body for
    your convenience. Static bodies can be moved, but there is a
    performance penalty as the collision information is recalculated.
    There is no penalty for having multiple static bodies, and it can be
    useful for simplifying your code by allowing different parts of your
    static geometry to be initialized or moved separately.
    """

    _pickle_args = "mass", "moment", "body_type"
    _pickle_kwargs = [
        "force",
        "angle",
        "position",
        "center_of_gravity",
        "velocity",
        "angular_velocity",
        "torque",
    ]
    _pickle_meta_hide = {
        "_cffi_ref",
        "_constraints",
        "_nursery",
        "_shapes",
        "_space",
        "_position_func",
        "_position_func_base",
        "_velocity_func",
        "_velocity_func_base",
        "shapes",
        "constraints",
        "is_sleeping",
    }
    _init_kwargs = {*_pickle_args, *_pickle_kwargs, "space"}
    _position_func_base: Optional[PositionFunc] = None  # For pickle
    _velocity_func_base: Optional[VelocityFunc] = None  # For pickle
    _id_counter = 1

    #
    # Properties and static methods
    #
    @staticmethod
    def update_velocity(
        body: "Body", gravity: VecLike, damping: float, dt: float
    ) -> None:
        """Default rigid body velocity integration function.

        Updates the velocity of the body using Euler integration.
        """
        lib.cpBodyUpdateVelocity(body._cffi_ref, gravity, damping, dt)

    @staticmethod
    def update_position(body: "Body", dt: float) -> None:
        """Default rigid body position integration function.

        Updates the position of the body using Euler integration. Unlike the
        velocity function, it's unlikely you'll want to override this
        function. If you do, make sure you understand it's source code
        (in Chipmunk) as it's an important part of the collision/joint
        correction process.
        """
        lib.cpBodyUpdatePosition(body._cffi_ref, dt)

    mass: float
    mass = property(  # type: ignore
        lambda self: lib.cpBodyGetMass(self._cffi_ref),
        lambda self, mass: void(lib.cpBodySetMass(self._cffi_ref, mass)),
        doc="""Mass of the body.""",
    )
    moment: float
    moment = property(  # type: ignore
        lambda self: lib.cpBodyGetMoment(self._cffi_ref),
        lambda self, moment: void(lib.cpBodySetMoment(self._cffi_ref, moment)),
        doc="""Moment of inertia (MoI or sometimes just moment) of the body.
    
        The moment is like the rotational mass of a body.
        """,
    )
    position: Vec2d
    position = property(  # type: ignore
        lambda self: vec2d_from_cffi(lib.cpBodyGetPosition(self._cffi_ref)),
        lambda self, pos: void(lib.cpBodySetPosition(self._cffi_ref, pos)),
        doc="""Position of the body.
    
        When changing the position you may also want to call
        :py:func:`Space.reindex_shapes_for_body` to update the collision 
        detection information for the attached shapes if plan to make any 
        queries against the space.""",
    )
    center_of_gravity: Vec2d
    center_of_gravity = property(  # type: ignore
        lambda self: vec2d_from_cffi(lib.cpBodyGetCenterOfGravity(self._cffi_ref)),
        lambda self, cog: void(lib.cpBodySetCenterOfGravity(self._cffi_ref, cog)),
        doc="""Location of the center of gravity in body local coordinates.
    
        The default value is (0, 0), meaning the center of gravity is the
        same as the position of the body.
        """,
    )
    velocity: Vec2d
    velocity = property(  # type: ignore
        lambda self: vec2d_from_cffi(lib.cpBodyGetVelocity(self._cffi_ref)),
        lambda self, vel: void(lib.cpBodySetVelocity(self._cffi_ref, vel)),
        doc="""Linear velocity of the center of gravity of the body.""",
    )
    force: Vec2d
    force = property(  # type: ignore
        lambda self: vec2d_from_cffi(lib.cpBodyGetForce(self._cffi_ref)),
        lambda self, f: void(lib.cpBodySetForce(self._cffi_ref, f)),
        doc="""Force applied to the center of gravity of the body.
    
        This value is reset for every time step. Note that this is not the 
        total of forces acting on the body (such as from collisions), but the 
        force applied manually from the apply force functions.""",
    )
    angle: float
    angle = property(  # type: ignore
        lambda self: degrees(lib.cpBodyGetAngle(self._cffi_ref)),
        lambda self, angle: void(lib.cpBodySetAngle(self._cffi_ref, radians(angle))),
        doc="""Rotation of the body in radians.
    
        When changing the rotation you may also want to call
        :py:func:`Space.reindex_shapes_for_body` to update the collision 
        detection information for the attached shapes if plan to make any 
        queries against the space. A body rotates around its center of gravity, 
        not its position.

        .. Note::
            If you get small/no changes to the angle when for example a
            ball is "rolling" down a slope it might be because the Circle shape
            attached to the body or the slope shape does not have any friction
            set.""",
    )
    angular_velocity: float
    angular_velocity = property(  # type: ignore
        lambda self: degrees(lib.cpBodyGetAngularVelocity(self._cffi_ref)),
        lambda self, w: void(lib.cpBodySetAngularVelocity(self._cffi_ref, radians(w))),
        doc="""The angular velocity of the body in radians per second.""",
    )
    torque: float
    torque = property(  # type: ignore
        lambda self: lib.cpBodyGetTorque(self._cffi_ref),
        lambda self, t: void(lib.cpBodySetTorque(self._cffi_ref, t)),
        doc="""The torque applied to the body.

        This value is reset for every time step.""",
    )
    rotation_vector: float
    rotation_vector = property(  # type: ignore
        lambda self: vec2d_from_cffi(lib.cpBodyGetRotation(self._cffi_ref)),
        doc="""The rotation vector for the body.""",
    )
    body_type: int
    body_type = property(  # type: ignore
        lambda self: lib.cpBodyGetType(self._cffi_ref),
        lambda self, body_type: void(lib.cpBodySetType(self._cffi_ref, body_type)),
        doc="""The type of a body (:py:const:`Body.DYNAMIC`, 
        :py:const:`Body.KINEMATIC` or :py:const:`Body.STATIC`).

        When changing an body to a dynamic body, the mass and moment of
        inertia are recalculated from the shapes added to the body. Custom
        calculated moments of inertia are not preserved when changing types.
        This function cannot be called directly in a collision callback.
        """,
    )
    velocity_func: Callable[["Body", VecLike, float, float], None]
    velocity_func = property(  # type: ignore
        fset=lambda self, fn: void(self._set_velocity_func(fn)),
        doc="""The velocity callback function. 
        
        The velocity callback function is called each time step, and can be 
        used to set a body's velocity.

            ``func(body : Body, gravity, damping, dt)``

        There are many cases when this can be useful. One example is individual 
        gravity for some bodies, and another is to limit the velocity which is 
        useful to prevent tunneling. 
        
        Example of a callback that sets gravity to zero for a object.

        >>> import pymunk
        >>> space = easymunk.Space()
        >>> space.gravity = 0, 10
        >>> body = easymunk.Body(1,2)
        >>> space.add(body)
        >>> def zero_gravity(body, gravity, damping, dt):
        ...     easymunk.Body.update_velocity(body, (0,0), damping, dt)
        ... 
        >>> body.velocity_func = zero_gravity
        >>> space.step(1)
        >>> space.step(1)
        >>> print(body.position, body.velocity)
        Vec2d(0.0, 0.0) Vec2d(0.0, 0.0)

        Example of a callback that limits the velocity:

        >>> import pymunk
        >>> body = easymunk.Body(1,2)
        >>> def limit_velocity(body, gravity, damping, dt):
        ...     max_velocity = 1000
        ...     easymunk.Body.update_velocity(body, gravity, damping, dt)
        ...     l = body.velocity.length
        ...     if l > max_velocity:
        ...         scale = max_velocity / l
        ...         body.velocity = body.velocity * scale
        ...
        >>> body.velocity_func = limit_velocity

        """,
    )
    position_func: Callable[["Body", float], None]
    position_func = property(  # type: ignore
        fset=lambda self, fn: void(self._set_position_func(fn)),
        doc="""The position callback function. 
            
        The position callback function is called each time step and can be 
        used to update the body's position.

            ``func(body, dt) -> None``
        """,
    )

    @property
    def is_sleeping(self) -> bool:
        """Returns true if the body is sleeping."""
        return bool(lib.cpBodyIsSleeping(self._cffi_ref))

    @property
    def space(self) -> Optional["Space"]:
        """Get the :py:class:`Space` that the body has been added to (or
        None)."""
        if self._space is not None:
            return py_space(self._space)
        else:
            return None

    @sk.lazy
    def constraints(self) -> Constraints:
        """Get the constraints this body is attached to.

        The body only keeps a weak reference to the constraints and a
        live body wont prevent GC of the attached constraints"""
        return Constraints(self, self._constraints)

    @sk.lazy
    def shapes(self) -> Shapes:
        """Get the shapes attached to this body.

        The body only keeps a weak reference to the shapes and a live
        body wont prevent GC of the attached shapes"""
        return Shapes(self, self._shapes)

    @property
    def arbiters(self) -> Set[Arbiter]:
        """
        Return list of arbiters on this body.
        """
        res: Set[Arbiter] = set()
        self.each_arbiter(res.add)
        return res

    #
    # Physical quantities
    #
    @property
    def kinetic_energy(self) -> float:
        """
        Kinetic energy for angular and linear components.
        """
        # todo: use ffi method?
        v2 = self.velocity.dot(self.velocity)
        w2 = self.angular_velocity * self.angular_velocity
        return 0.5 * (
            (self.mass * v2 if v2 else 0.0) + (self.moment * w2 if w2 else 0.0)
        )

    @property
    def gravitational_energy(self) -> float:
        """
        Potential energy due to gravity. Zero if not included in a space..
        """
        if self.space is None:
            return 0.0
        gravity = self.space.gravity
        return -self.mass * self.position.dot(gravity)

    @property
    def linear_momentum(self) -> Vec2d:
        """
        Body's linear momentum (mass times velocity).
        """
        return self.mass * self.velocity

    @property
    def angular_momentum(self) -> float:
        """
        Angular momentum around the center of mass.
        """
        return self.moment * self.angular_velocity

    @property
    def density(self) -> float:
        """
        Overall density of body. If a density value is assigned, it fixes the
        density of all shapes in body.
        """
        mass = 0.0
        area = 0.0
        for s in self.shapes:
            mass += s.mass
            area += s.area
        return mass / area

    @density.setter
    def density(self, value):
        self.each_shape(density=value)
        self.mass = sum(s.mass for s in self.shapes)

    @property
    def elasticity(self) -> Optional[float]:
        """
        Get/Set elasticity of shapes connected to body.

        Elasticity is None if body has not connected shapes or if shapes have
        different elasticities.
        """
        try:
            value, *other = set(s.elasticity for s in self.shapes)
        except IndexError:
            return None
        else:
            return None if other else value

    @elasticity.setter
    def elasticity(self, value):
        self.each_shape(elasticity=value)

    @property
    def friction(self) -> Optional[float]:
        """
        Get/Set friction of shapes connected to body.

        friction is None if body has not connected shapes or if shapes have
        different friction coefficients.
        """
        try:
            value, *other = set(s.friction for s in self.shapes)
        except IndexError:
            return None
        else:
            return None if other else value

    @friction.setter
    def friction(self, value):
        self.each_shape(friction=value)

    @property
    def _id(self) -> int:
        """Unique id of the Body

        .. note::
            Experimental API. Likely to change in future major, minor orpoint
            releases.
        """
        return int(ffi.cast("int", lib.cpBodyGetUserData(self._cffi_ref)))

    def __init__(
        self,
        mass: float = 0,
        moment: float = 0,
        body_type: BodyType = DYNAMIC,
        *,
        space=None,
        **kwargs,
    ) -> None:
        """Create a new Body

        Mass and moment are ignored when body_type is KINEMATIC or STATIC.

        Guessing the mass for a body is usually fine, but guessing a moment
        of inertia can lead to a very poor simulation so it's recommended to
        use Chipmunk's moment calculations to estimate the moment for you.

        There are two ways to set up a dynamic body. The easiest option is to
        create a body with a mass and moment of 0, and set the mass or
        density of each collision shape added to the body. Chipmunk will
        automatically calculate the mass, moment of inertia, and center of
        gravity for you. This is probably preferred in most cases. Note that
        these will only be correctly calculated **after** the body and shape are
        added to a space.

        The other option is to set the mass of the body when it's created,
        and leave the mass of the shapes added to it as 0.0. This approach is
        more flexible, but is not as easy to use. Don't set the mass of both
        the body and the shapes. If you do so, it will recalculate and
        overwrite your custom mass value when the shapes are added to the body.
        """

        body_type = BODY_TYPES.get(body_type, body_type)
        if body_type == Body.DYNAMIC:
            self._cffi_ref = ffi.gc(lib.cpBodyNew(mass, moment), cffi_free_body)
        elif body_type == Body.KINEMATIC:
            self._cffi_ref = ffi.gc(lib.cpBodyNewKinematic(), cffi_free_body)
        elif body_type == Body.STATIC:
            self._cffi_ref = ffi.gc(lib.cpBodyNewStatic(), cffi_free_body)
        else:
            raise ValueError(f"invalid body type: {body_type!r}")

        # To prevent the gc to collect the callbacks.
        self._position_func = None
        self._velocity_func = None

        # For pickle
        self._position_func_base = None
        self._velocity_func_base = None

        # Weak refs to space, shapes and constraints (if any)
        self._space: Optional["Space"] = None
        self._constraints: WeakSet["Constraint"] = WeakSet()
        self._shapes: WeakSet["Shape"] = WeakSet()

        # Keep references before adding objects to space
        self._nursery: List["Shape"] = []

        self._set_id()
        init_attributes(self, self._init_kwargs, kwargs)
        if space is not None:
            space.add(self)

    def __getstate__(self):
        args, meta = super().__getstate__()

        if self._position_func is not None:
            meta["position_func"] = self._position_func_base
        if self._velocity_func is not None:
            meta["velocity_func"] = self._velocity_func_base

        meta["$shapes"] = {s.copy() for s in list(self._shapes)}
        return args, meta

    def __setstate__(self, state):
        args, meta = state
        shapes = meta.pop("$shapes")
        super().__setstate__((args, meta))

        for shape in shapes:
            shape.body = self

    def __repr__(self) -> str:
        if self.body_type == Body.DYNAMIC:
            return "Body(%r, %r, Body.DYNAMIC)" % (self.mass, self.moment)
        elif self.body_type == Body.KINEMATIC:
            return "Body(Body.KINEMATIC)"
        else:
            return "Body(Body.STATIC)"

    def _iter_bounding_boxes(self) -> Iterator["BB"]:
        for s in self._shapes:
            if not s.sensor:
                yield s.bb

    def _iter_constraints(self) -> Iterator["Constraint"]:
        yield from self._constraints

    def _iter_shapes(self) -> Iterator["Shape"]:
        yield from self._shapes

    def _iter_bodies(self) -> Iterator["Body"]:
        yield self

    def _set_id(self) -> None:
        lib.cpBodySetUserData(
            self._cffi_ref, ffi.cast("cpDataPointer", Body._id_counter)
        )
        Body._id_counter += 1

    def _create_shape(self, cls, args, kwargs):
        shape = cls(*args, body=self, **kwargs)
        if self.space is not None:
            self.space.add(shape)
        return shape

    def _set_velocity_func(self, func: VelocityFunc) -> None:
        @ffi.callback("cpBodyVelocityFunc")
        def _impl(_: ffi.CData, gravity: ffi.CData, damping: float, dt: float) -> None:
            func(self, Vec2d(gravity.x, gravity.y), damping, dt)

        self._velocity_func_base = func
        self._velocity_func = _impl
        lib.cpBodySetVelocityUpdateFunc(self._cffi_ref, _impl)

    def _set_position_func(self, func: Callable[["Body", float], None]) -> None:
        @ffi.callback("cpBodyPositionFunc")
        def _impl(_: ffi.CData, dt: float) -> None:
            return func(self, dt)

        self._position_func_base = func
        self._position_func = _impl
        lib.cpBodySetPositionUpdateFunc(self._cffi_ref, _impl)

    def cache_bb(self):
        for s in self.shapes:
            s.cache_bb()
        return self.bb

    def apply_force(self: B, force) -> B:
        """
        Apply force to the center of mass (does not produce any resulting torque).
        """
        self.force += force
        return self

    def apply_torque(self: B, torque) -> B:
        """
        Apply toque to the center of mass (does not produce any resulting force).
        """
        self.torque += torque
        return self

    def apply_force_at_world_point(self: B, force: VecLike, point: VecLike) -> B:
        """Add the force force to body as if applied from the world point.

        People are sometimes confused by the difference between a force and
        an impulse. An impulse is a very large force applied over a very
        short period of time. Some examples are a ball hitting a wall or
        cannon firing. Chipmunk treats impulses as if they occur
        instantaneously by adding directly to the velocity of an object.
        Both impulses and forces are affected the mass of an object. Doubling
        the mass of the object will halve the effect.
        """
        lib.cpBodyApplyForceAtWorldPoint(self._cffi_ref, force, point)
        return self

    def apply_force_at_local_point(
        self: B, force: VecLike, point: VecLike = (0, 0)
    ) -> B:
        """Add the local force force to body as if applied from the body
        local point.
        """
        lib.cpBodyApplyForceAtLocalPoint(self._cffi_ref, force, point)
        return self

    def apply_impulse_at_world_point(self: B, impulse: VecLike, point: VecLike) -> B:
        """Add the impulse impulse to body as if applied from the world point."""
        lib.cpBodyApplyImpulseAtWorldPoint(self._cffi_ref, impulse, point)
        return self

    def apply_impulse_at_local_point(
        self: B, impulse: VecLike, point: VecLike = (0, 0)
    ) -> B:
        """Add the local impulse impulse to body as if applied from the body
        local point.
        """
        lib.cpBodyApplyImpulseAtLocalPoint(self._cffi_ref, impulse, point)
        return self

    def activate(self: B) -> B:
        """Reset the idle timer on a body.

        If it was sleeping, wake it and any other bodies it was touching.
        """
        lib.cpBodyActivate(self._cffi_ref)
        return self

    def sleep(self: B) -> B:
        """Forces a body to fall asleep immediately even if it's in midair.

        Cannot be called from a callback.
        """
        if self._space is None:
            raise Exception("Body not added to space")
        lib.cpBodySleep(self._cffi_ref)
        return self

    def sleep_with_group(self: B, body: "Body") -> B:
        """Force a body to fall asleep immediately along with other bodies
        in a group.

        When objects in Pymunk sleep, they sleep as a group of all objects
        that are touching or jointed together. When an object is woken up,
        all of the objects in its group are woken up.
        :py:func:`Body.sleep_with_group` allows you group sleeping objects
        together. It acts identically to :py:func:`Body.sleep` if you pass
        None as group by starting a new group. If you pass a sleeping body
        for group, body will be awoken when group is awoken. You can use this
        to initialize levels and start stacks of objects in a pre-sleeping
        state.
        """
        if self._space is None:
            raise Exception("Body not added to space")
        lib.cpBodySleepWithGroup(self._cffi_ref, body._cffi_ref)
        return self

    def each_arbiter(
        self: B, func: Callable[..., None] = set_attrs, *args: Any, **kwargs: Any
    ) -> B:
        """Run func on each of the arbiters on this body.

            ``func(arbiter, *args, **kwargs) -> None``

            Callback Parameters
                arbiter : :py:class:`Arbiter`
                    The Arbiter
                args
                    Optional parameters passed to the callback function.
                kwargs
                    Optional keyword parameters passed on to the callback function.

        The default function is :py:func:`set_attrs` and simply set attributes
        passed as keyword parameters.

        .. warning::

            Do not hold on to the Arbiter after the callback!
        """

        @ffi.callback("cpBodyArbiterIteratorFunc")
        def cf(_body, arb, _) -> None:
            if self._space is None:
                raise ValueError("Body does not belong to any space")
            arbiter = Arbiter(arb, self._space)
            func(arbiter, *args, **kwargs)

        lib.cpBodyEachArbiter(self._cffi_ref, cf, ffi.new_handle(self))
        return self

    def _each(self: B, _col, _fn, args, kwargs) -> B:
        for item in _col:
            _fn(item, *args, **kwargs)
        return self

    def each_constraint(self: B, _fn=set_attrs, *args, **kwargs) -> B:
        """Run func on each of the constraints on this body.

            ``func(constraint, *args, **kwargs) -> None``

            Callback Parameters
                constraint : :py:class:`Constraint`
                    The Constraint
                args
                    Optional parameters passed to the callback function.
                kwargs
                    Optional keyword parameters passed on to the callback function.

        The default function is :py:func:`set_attrs` and simply set attributes
        passed as keyword parameters.
        """
        return self._each(self._constraints, _fn, args, kwargs)

    def each_shape(self: B, _fn=set_attrs, *args, **kwargs) -> B:
        """Run func on each of the shapes on this body.

            ``func(shape, *args, **kwargs) -> None``

            Callback Parameters
                shape : :py:class:`Shape`
                    The Shape
                args
                    Optional parameters passed to the callback function.
                kwargs
                    Optional keyword parameters passed on to the callback function.

        The default function is :py:func:`set_attrs` and simply set attributes
        passed as keyword parameters.
        """
        return self._each(self._shapes, _fn, args, kwargs)

    def local_to_world(self, v: VecLike) -> Vec2d:
        """Convert body local coordinates to world space coordinates

        Many things are defined in coordinates local to a body meaning that
        the (0,0) is at the center of gravity of the body and the axis rotate
        along with the body.

        :param v: Vector in body local coordinates
        """
        v2 = lib.cpBodyLocalToWorld(self._cffi_ref, v)
        return Vec2d(v2.x, v2.y)

    def world_to_local(self, v: VecLike) -> Vec2d:
        """Convert world space coordinates to body local coordinates

        :param v: Vector in world space coordinates
        """
        v2 = lib.cpBodyWorldToLocal(self._cffi_ref, v)
        return Vec2d(v2.x, v2.y)

    def velocity_at_world_point(self, point: VecLike) -> Vec2d:
        """Get the absolute velocity of the rigid body at the given world
        point

        It's often useful to know the absolute velocity of a point on the
        surface of a body since the angular velocity affects everything
        except the center of gravity.
        """
        v = lib.cpBodyGetVelocityAtWorldPoint(self._cffi_ref, point)
        return Vec2d(v.x, v.y)

    def velocity_at_local_point(self, point: VecLike) -> Vec2d:
        """Get the absolute velocity of the rigid body at the given body
        local point
        """
        v = lib.cpBodyGetVelocityAtLocalPoint(self._cffi_ref, point)
        return Vec2d(v.x, v.y)

    def reindex_shapes(self: B) -> B:
        """Reindex all shapes in body.

        It is a NO-OP if body is not in a space."""
        space = self.space
        if space is not None:
            space.reindex_shapes_for_body(self)
        return self

    def copy(self):
        return self.prepare()

    def prepare(self, **kwargs):
        state = self.__getstate__()
        new = object.__new__(type(self))
        new.__setstate__(state)

        for shape in self.shapes:
            sp = shape.prepare(body=new)
            new._nursery.append(sp)

        for k, v in kwargs.items():
            setattr(new, k, v)

        return new

    #
    # Transforms
    #
    def update(self: B, **kwargs) -> B:
        """
        Update variables in body.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def axis(self, obj: Union[str, VecLike]) -> Vec2d:
        """
        Return axis from string
        """
        if not isinstance(obj, str):
            return Vec2d(*obj)
        if obj == "middle":
            print("center", self.bb.center())
            return self.cache_bb().center() - self.position
        elif obj == "pos":
            return Vec2d(0.0, 0.0)
        raise NotImplementedError

    def rotate(self: B, angle, axis=None) -> B:
        """
        Rotate body by angle.
        """
        # if not angle:
        #     return self

        print(self.position, self.cache_bb())
        # print(self.center_of_gravity, self.position)
        if axis is not None:
            center = self.axis(axis)
            print("center:", center)
            self.position += center.rotated(angle) - center
        self.angle += angle
        return self

    def move(self: B, x_or_vec, y=None) -> B:
        """
        Rotate body by angle.
        """
        vec = x_or_vec if y is None else (x_or_vec, y)
        self.position += vec
        return self

    def fuse_with(self: B, other: "Body") -> B:
        """
        Fuse shapes of other objects into self.
        """
        from .shapes import Circle, Poly

        for shape in other.shapes:
            if isinstance(shape, Circle):
                offset = self.world_to_local(other.local_to_world(shape.offset))
                shape = Circle(shape.radius, offset, self)
            elif isinstance(shape, Poly):
                vertices = [
                    self.world_to_local(v) for v in shape.get_vertices(world=True)
                ]
                shape = Poly(vertices, radius=shape.radius, body=self)
            else:
                raise NotImplementedError

            if self.space is not None:
                self.space.add(shape)
            else:
                self._nursery.append(shape)

        self.mass += other.mass
        self.moment += other.moment  # FIXME: that is not how moments work ;)
        return self

    def _mirror(self, value) -> Tuple[Vec2d, Vec2d]:
        if value == "left":
            bb = self.cache_bb()
            return Vec2d(bb.left, 0), Vec2d(1, 0)
        raise NotImplementedError

    def flip(self: B, axis) -> B:
        from .shapes import Circle, Poly

        p0, n = self._mirror(axis)
        t = n.perpendicular()

        def pt_transform(p: Vec2d):
            d = p - p0
            return p0 + d - 2 * t * p.dot(t)

        for shape in self.shapes:
            if isinstance(shape, Circle):
                shape.offset = pt_transform(shape.offset)
            elif isinstance(shape, Poly):
                vertices = [pt_transform(v) for v in shape.get_vertices()]
                shape.set_vertices(vertices)
            else:
                raise NotImplementedError

        return self

    def scale(self: B, scale, axis=None) -> B:
        from .shapes import Circle, Poly

        if axis is not None:
            raise NotImplementedError

        for shape in self.shapes:
            if isinstance(shape, Circle):
                shape.radius *= scale
            elif isinstance(shape, Poly):
                vertices = [scale * v for v in shape.get_vertices()]
                shape.set_vertices(vertices)
            else:
                raise NotImplementedError

        return self


#
# Specialized bodies
#
class BodyShape(Body):
    """
    Base class for bodies with a single shape.
    """

    shape: "Shape"

    # Properties
    radius: float = sk.delegate_to("shape", mutable=True)
    area: float = sk.delegate_to("shape")
    collision_type: int = sk.delegate_to("shape", mutable=True)
    filter: "ShapeFilter" = sk.delegate_to("shape", mutable=True)
    elasticity: float = sk.delegate_to("shape", mutable=True)
    friction: float = sk.delegate_to("shape", mutable=True)
    surface_velocity: Vec2d = sk.delegate_to("shape", mutable=True)
    bb: "BB" = sk.delegate_to("shape")

    # Methods
    point_query = sk.delegate_to("shape")
    segment_query = sk.delegate_to("shape")
    shapes_collide = sk.delegate_to("shape")
    body: "Body" = property(lambda self: self)
    radius_of_gyration_sqr = sk.delegate_to("shape")

    def _extract_options(self, kwargs):
        opts = {}
        for k in self._init_kwargs:
            if k in kwargs:
                opts[k] = kwargs.pop(k)
        return opts

    def _post_init(self):
        if (
            self.body_type == Body.DYNAMIC
            and self.mass == 0.0
            and self.space is not None
        ):
            self.mass = self.shape.area
            self.moment = self.mass * self.radius_of_gyration_sqr()


class CircleBody(BodyShape, Body):
    """
    A body attached to a single circular shape.
    """

    offset: Vec2d = sk.delegate_to("shape", mutable=True)

    def __init__(self, radius, *args, offset=(0, 0), **kwargs):
        super().__init__(*args, **self._extract_options(kwargs))
        self.shape: "Circle" = Circle(radius, offset=offset, body=self, **kwargs)
        self._post_init()


class PolyBody(BodyShape, Body):
    """
    A body attached to a single polygonal shape.
    """

    get_vertices = sk.delegate_to("shape")
    set_vertices = sk.delegate_to("shape")

    @classmethod
    def _new_from_shape_factory(cls, mk_shape, *args, **kwargs):
        new = object.__new__(PolyBody)
        Body.__init__(new, *args, **new._extract_options(kwargs))
        # pprint(new.__dict__)
        new.shape = mk_shape(new, **kwargs)
        new._post_init()
        return new

    @classmethod
    def new_box(cls, size: VecLike = (10, 10), *args, radius: float = 0.0, **kwargs):
        mk_box = lambda body, **opts: Poly.new_box(size, radius, body, **opts)
        return cls._new_from_shape_factory(mk_box, *args, **kwargs)

    @classmethod
    def new_box_bb(cls, bb: "BB", *args, radius: float = 0.0, **kwargs):
        mk_box = lambda body, **opts: Poly.new_box_bb(bb, radius, body, **opts)
        return cls._new_from_shape_factory(mk_box, *args, **kwargs)

    @classmethod
    def new_regular_poly(
        cls,
        n: int,
        size: float,
        radius: float = 0.0,
        *args,
        angle: float = 0.0,
        offset: VecLike = (0, 0),
        **kwargs,
    ):
        mk_box = lambda body, **opts: Poly.new_regular_poly(
            n, size, radius, body, angle=angle, offset=offset, **opts
        )
        return cls._new_from_shape_factory(mk_box, *args, **kwargs)

    def __init__(self, vertices, *args, radius=0, **kwargs):
        super().__init__(*args, **self._extract_options(kwargs))
        self.shape: "Poly" = Poly(vertices, radius=radius, body=self, **kwargs)
        self._post_init()


class SegmentBody(BodyShape, Body):
    """
    A body attached to a single circular shape.
    """

    a: Vec2d = sk.delegate_to("shape", mutable=True)
    b: Vec2d = sk.delegate_to("shape", mutable=True)

    def __init__(self, a, b, *args, radius=0, **kwargs):
        super().__init__(*args, **self._extract_options(kwargs))
        self.shape: "Segment" = Segment(a, b, radius, self, **kwargs)
        self._post_init()


#
# Utility functions
#
def cffi_free_body(cp_body):
    logging.debug("bodyfree start %s", cp_body)
    cp_shapes = []
    cp_constraints = []

    @ffi.callback("cpBodyShapeIteratorFunc")
    def cf1(_, shape, __):
        cp_shapes.append(shape)

    @ffi.callback("cpBodyConstraintIteratorFunc")
    def cf2(_, constraint, __):
        cp_constraints.append(constraint)

    lib.cpBodyEachShape(cp_body, cf1, ffi.NULL)
    for cp_shape in cp_shapes:
        logging.debug("free %s %s", cp_body, cp_shape)
        cp_space = lib.cpShapeGetSpace(cp_shape)
        if cp_space != ffi.NULL:
            lib.cpSpaceRemoveShape(cp_space, cp_shape)

    lib.cpBodyEachConstraint(cp_body, cf2, ffi.NULL)
    for cp_constraint in cp_constraints:
        logging.debug("free %s %s", cp_body, cp_constraint)
        cp_space = lib.cpConstraintGetSpace(cp_constraint)
        if cp_space != ffi.NULL:
            lib.cpSpaceRemoveConstraint(cp_space, cp_constraint)

    cp_space = lib.cpBodyGetSpace(cp_body)
    if cp_space != ffi.NULL:
        lib.cpSpaceRemoveBody(cp_space, cp_body)

    logging.debug("bodyfree free %s", cp_body)
    lib.cpBodyFree(cp_body)
