__docformat__ = "reStructuredText"

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Set,
    Tuple,
    Iterable,
    TypeVar,
    List,
)
from weakref import WeakSet

if TYPE_CHECKING:
    from .space import Space
    from .constraints import Constraint
    from .shapes import Shape
    from .bb import BB

from ._chipmunk_cffi import ffi, lib
from ._mixins import (
    PickleMixin,
    _State,
    TypingAttrMixing,
    HasBBMixin,
    FilterElementsMixin,
)
from .arbiter import Arbiter
from .vec2d import Vec2d, VecLike, vec2d_from_cffi
from .util import void, set_attrs, py_space, init_attributes

T = TypeVar("T")
_BodyType = int
_PositionFunc = Callable[["Body", float], None]
_VelocityFunc = Callable[["Body", Vec2d, float, float], None]


class Body(PickleMixin, TypingAttrMixing, HasBBMixin, FilterElementsMixin):
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

    DYNAMIC = lib.CP_BODY_TYPE_DYNAMIC
    """Dynamic bodies are the default body type.

    They react to collisions,
    are affected by forces and gravity, and have a finite amount of mass.
    These are the type of bodies that you want the physics engine to
    simulate for you. Dynamic bodies interact with all types of bodies
    and can generate collision callbacks.
    """

    KINEMATIC = lib.CP_BODY_TYPE_KINEMATIC
    """Kinematic bodies are bodies that are controlled from your code
    instead of inside the physics engine.

    They arent affected by gravity and they have an infinite amount of mass
    so they don't react to collisions or forces with other bodies. Kinematic
    bodies are controlled by setting their velocity, which will cause them
    to move. Good examples of kinematic bodies might include things like
    moving platforms. Objects that are touching or jointed to a kinematic
    body are never allowed to fall asleep.
    """

    STATIC = lib.CP_BODY_TYPE_STATIC
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

    _pickle_attrs_init = PickleMixin._pickle_attrs_init + [
        "mass",
        "moment",
        "body_type",
    ]
    _pickle_attrs_general = PickleMixin._pickle_attrs_general + [
        "force",
        "angle",
        "position",
        "center_of_gravity",
        "velocity",
        "angular_velocity",
        "torque",
    ]
    _pickle_attrs_skip = PickleMixin._pickle_attrs_skip + [
        "is_sleeping",
        "_velocity_func",
        "_position_func",
    ]
    _init_kwargs = {
        "position",
        "velocity",
        "center_of_gravity",
        "angle",
        "angular_velocity",
    }

    _position_func_base: Optional[_PositionFunc] = None  # For pickle
    _velocity_func_base: Optional[_VelocityFunc] = None  # For pickle
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
        lib.cpBodyUpdateVelocity(body._body, gravity, damping, dt)

    @staticmethod
    def update_position(body: "Body", dt: float) -> None:
        """Default rigid body position integration function.

        Updates the position of the body using Euler integration. Unlike the
        velocity function, it's unlikely you'll want to override this
        function. If you do, make sure you understand it's source code
        (in Chipmunk) as it's an important part of the collision/joint
        correction process.
        """
        lib.cpBodyUpdatePosition(body._body, dt)

    mass: float = property(
        lambda self: lib.cpBodyGetMass(self._body),
        lambda self, mass: void(lib.cpBodySetMass(self._body, mass)),
        doc="""Mass of the body.""",
    )
    moment: float = property(
        lambda self: lib.cpBodyGetMoment(self._body),
        lambda self, moment: void(lib.cpBodySetMoment(self._body, moment)),
        doc="""Moment of inertia (MoI or sometimes just moment) of the body.
    
        The moment is like the rotational mass of a body.
        """,
    )
    position: Vec2d = property(
        lambda self: vec2d_from_cffi(lib.cpBodyGetPosition(self._body)),
        lambda self, pos: void(lib.cpBodySetPosition(self._body, pos)),
        doc="""Position of the body.
    
        When changing the position you may also want to call
        :py:func:`Space.reindex_shapes_for_body` to update the collision 
        detection information for the attached shapes if plan to make any 
        queries against the space.""",
    )
    center_of_gravity: Vec2d = property(
        lambda self: vec2d_from_cffi(lib.cpBodyGetCenterOfGravity(self._body)),
        lambda self, cog: void(lib.cpBodySetCenterOfGravity(self._body, cog)),
        doc="""Location of the center of gravity in body local coordinates.
    
        The default value is (0, 0), meaning the center of gravity is the
        same as the position of the body.
        """,
    )
    velocity: Vec2d = property(
        lambda self: vec2d_from_cffi(lib.cpBodyGetVelocity(self._body)),
        lambda self, vel: void(lib.cpBodySetVelocity(self._body, vel)),
        doc="""Linear velocity of the center of gravity of the body.""",
    )
    force: Vec2d = property(
        lambda self: vec2d_from_cffi(lib.cpBodyGetForce(self._body)),
        lambda self, f: void(lib.cpBodySetForce(self._body, f)),
        doc="""Force applied to the center of gravity of the body.
    
        This value is reset for every time step. Note that this is not the 
        total of forces acting on the body (such as from collisions), but the 
        force applied manually from the apply force functions.""",
    )
    angle: float = property(
        lambda self: lib.cpBodyGetAngle(self._body),
        lambda self, angle: void(lib.cpBodySetAngle(self._body, angle)),
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
    angular_velocity: float = property(
        lambda self: lib.cpBodyGetAngularVelocity(self._body),
        lambda self, w: void(lib.cpBodySetAngularVelocity(self._body, w)),
        doc="""The angular velocity of the body in radians per second.""",
    )
    torque: float = property(
        lambda self: lib.cpBodyGetTorque(self._body),
        lambda self, t: void(lib.cpBodySetTorque(self._body, t)),
        doc="""The torque applied to the body.

        This value is reset for every time step.""",
    )
    rotation_vector: float = property(
        lambda self: vec2d_from_cffi(lib.cpBodyGetRotation(self._body)),
        doc="""The rotation vector for the body.""",
    )

    @property
    def is_sleeping(self) -> bool:
        """Returns true if the body is sleeping."""
        return bool(lib.cpBodyIsSleeping(self._body))

    body_type = property(
        lambda self: lib.cpBodyGetType(self._body),
        lambda self, body_type: void(lib.cpBodySetType(self._body, body_type)),
        doc="""The type of a body (:py:const:`Body.DYNAMIC`, 
        :py:const:`Body.KINEMATIC` or :py:const:`Body.STATIC`).

        When changing an body to a dynamic body, the mass and moment of
        inertia are recalculated from the shapes added to the body. Custom
        calculated moments of inertia are not preserved when changing types.
        This function cannot be called directly in a collision callback.
        """,
    )

    @property
    def space(self) -> Optional["Space"]:
        """Get the :py:class:`Space` that the body has been added to (or
        None)."""
        if self._space is not None:
            return py_space(self._space)
        else:
            return None

    def _set_velocity_func(self, func: _VelocityFunc) -> None:
        @ffi.callback("cpBodyVelocityFunc")
        def _impl(_: ffi.CData, gravity: ffi.CData, damping: float, dt: float) -> None:
            func(self, Vec2d(gravity.x, gravity.y), damping, dt)

        self._velocity_func_base = func
        self._velocity_func = _impl
        lib.cpBodySetVelocityUpdateFunc(self._body, _impl)

    velocity_func = property(
        fset=_set_velocity_func,
        doc="""The velocity callback function. 
        
        The velocity callback function is called each time step, and can be 
        used to set a body's velocity.

            ``func(body : Body, gravity, damping, dt)``

        There are many cases when this can be useful. One example is individual 
        gravity for some bodies, and another is to limit the velocity which is 
        useful to prevent tunneling. 
        
        Example of a callback that sets gravity to zero for a object.

        >>> import pymunk
        >>> space = pymunk.Space()
        >>> space.gravity = 0, 10
        >>> body = pymunk.Body(1,2)
        >>> space.add(body)
        >>> def zero_gravity(body, gravity, damping, dt):
        ...     pymunk.Body.update_velocity(body, (0,0), damping, dt)
        ... 
        >>> body.velocity_func = zero_gravity
        >>> space.step(1)
        >>> space.step(1)
        >>> print(body.position, body.velocity)
        Vec2d(0.0, 0.0) Vec2d(0.0, 0.0)

        Example of a callback that limits the velocity:

        >>> import pymunk
        >>> body = pymunk.Body(1,2)
        >>> def limit_velocity(body, gravity, damping, dt):
        ...     max_velocity = 1000
        ...     pymunk.Body.update_velocity(body, gravity, damping, dt)
        ...     l = body.velocity.length
        ...     if l > max_velocity:
        ...         scale = max_velocity / l
        ...         body.velocity = body.velocity * scale
        ...
        >>> body.velocity_func = limit_velocity

        """,
    )

    def _set_position_func(self, func: Callable[["Body", float], None]) -> None:
        @ffi.callback("cpBodyPositionFunc")
        def _impl(_: ffi.CData, dt: float) -> None:
            return func(self, dt)

        self._position_func_base = func
        self._position_func = _impl
        lib.cpBodySetPositionUpdateFunc(self._body, _impl)

    position_func = property(
        fset=_set_position_func,
        doc="""The position callback function. 
            
        The position callback function is called each time step and can be 
        used to update the body's position.

            ``func(body, dt) -> None``
        """,
    )

    @property
    def constraints(self) -> Set["Constraint"]:
        """Get the constraints this body is attached to.

        The body only keeps a weak reference to the constraints and a
        live body wont prevent GC of the attached constraints"""
        return set(self._constraints)

    #
    # Bounding box and shape-related properties
    #
    _bodies: List["Body"] = property(lambda self: [self])
    _shapes: Set["Shape"] = set()
    _constraints: Set["Constraint"] = set()

    @property
    def shapes(self) -> Set["Shape"]:
        """Get the shapes attached to this body.

        The body only keeps a weak reference to the shapes and a live
        body wont prevent GC of the attached shapes"""
        return set(self._shapes)

    def _iter_bounding_boxes(self) -> Iterable["BB"]:
        for s in self._shapes:
            if not s.sensor:
                yield s.bb

    @property
    def arbiters(self) -> Set[Arbiter]:
        """
        Return list of arbiters on this body.
        """
        res = set()
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
        # return lib._cpBodyKineticEnergy(self._body)
        v2 = self.velocity.dot(self.velocity)
        w2 = self.angular_velocity * self.angular_velocity
        return 0.5 * (
            (self.mass * v2 if v2 else 0.0) + (self.moment * w2 if w2 else 0.0)
        )

    @property
    def gravitational_energy(self) -> float:
        """
        Potential energy due to gravity.
        """
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
        return int(ffi.cast("int", lib.cpBodyGetUserData(self._body)))

    def __init__(
        self,
        mass: float = 0,
        moment: float = 0,
        body_type: _BodyType = DYNAMIC,
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

        if body_type == Body.DYNAMIC:
            self._body = ffi.gc(lib.cpBodyNew(mass, moment), cffi_free_body)
        elif body_type == Body.KINEMATIC:
            self._body = ffi.gc(lib.cpBodyNewKinematic(), cffi_free_body)
        elif body_type == Body.STATIC:
            self._body = ffi.gc(lib.cpBodyNewStatic(), cffi_free_body)

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

        self._set_id()
        init_attributes(self, self._init_kwargs, kwargs)

    def __getstate__(self) -> _State:
        """Return the state of this object

        This method allows the usage of the :mod:`copy` and :mod:`pickle`
        modules with this class.
        """
        d = super(Body, self).__getstate__()

        d["special"].append(("is_sleeping", self.is_sleeping))
        d["special"].append(("_velocity_func", self._velocity_func_base))
        d["special"].append(("_position_func", self._position_func_base))

        return d

    def __setstate__(self, state: _State) -> None:
        """Unpack this object from a saved state.

        This method allows the usage of the :mod:`copy` and :mod:`pickle`
        modules with this class.
        """
        super(Body, self).__setstate__(state)

        for k, v in state["special"]:
            if k == "is_sleeping" and v:
                pass
            elif k == "_velocity_func" and v is not None:
                self.velocity_func = v
            elif k == "_position_func" and v is not None:
                self.position_func = v

    def __repr__(self) -> str:
        if self.body_type == Body.DYNAMIC:
            return "Body(%r, %r, Body.DYNAMIC)" % (self.mass, self.moment)
        elif self.body_type == Body.KINEMATIC:
            return "Body(Body.KINEMATIC)"
        else:
            return "Body(Body.STATIC)"

    def _set_id(self) -> None:
        lib.cpBodySetUserData(self._body, ffi.cast("cpDataPointer", Body._id_counter))
        Body._id_counter += 1

    def apply_force(self: T, force) -> T:
        """
        Apply force to the center of mass (does not produce any resulting torque).
        """
        self.force += force
        return self

    def apply_torque(self: T, torque) -> T:
        """
        Apply toque to the center of mass (does not produce any resulting force).
        """
        self.torque += torque
        return self

    def apply_force_at_world_point(self: T, force: VecLike, point: VecLike) -> T:
        """Add the force force to body as if applied from the world point.

        People are sometimes confused by the difference between a force and
        an impulse. An impulse is a very large force applied over a very
        short period of time. Some examples are a ball hitting a wall or
        cannon firing. Chipmunk treats impulses as if they occur
        instantaneously by adding directly to the velocity of an object.
        Both impulses and forces are affected the mass of an object. Doubling
        the mass of the object will halve the effect.
        """
        lib.cpBodyApplyForceAtWorldPoint(self._body, force, point)
        return self

    def apply_force_at_local_point(
        self: T, force: VecLike, point: VecLike = (0, 0)
    ) -> T:
        """Add the local force force to body as if applied from the body
        local point.
        """
        lib.cpBodyApplyForceAtLocalPoint(self._body, force, point)
        return self

    def apply_impulse_at_world_point(self: T, impulse: VecLike, point: VecLike) -> T:
        """Add the impulse impulse to body as if applied from the world point."""
        lib.cpBodyApplyImpulseAtWorldPoint(self._body, impulse, point)
        return self

    def apply_impulse_at_local_point(
        self: T, impulse: VecLike, point: VecLike = (0, 0)
    ) -> T:
        """Add the local impulse impulse to body as if applied from the body
        local point.
        """
        lib.cpBodyApplyImpulseAtLocalPoint(self._body, impulse, point)
        return self

    def activate(self: T) -> T:
        """Reset the idle timer on a body.

        If it was sleeping, wake it and any other bodies it was touching.
        """
        lib.cpBodyActivate(self._body)
        return self

    def sleep(self: T) -> T:
        """Forces a body to fall asleep immediately even if it's in midair.

        Cannot be called from a callback.
        """
        if self._space is None:
            raise Exception("Body not added to space")
        lib.cpBodySleep(self._body)
        return self

    def sleep_with_group(self: T, body: "Body") -> T:
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
        lib.cpBodySleepWithGroup(self._body, body._body)
        return self

    def each_arbiter(
        self: T, func: Callable[..., None] = set_attrs, *args: Any, **kwargs: Any
    ) -> T:
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
        def cf(_body: ffi.CData, _arbiter: ffi.CData, _data: ffi.CData) -> None:
            if self._space is None:
                raise ValueError("Body does not belong to any space")
            arbiter = Arbiter(_arbiter, self._space)
            func(arbiter, *args, **kwargs)

        lib.cpBodyEachArbiter(self._body, cf, ffi.new_handle(self))
        return self

    def _each(self: T, _col, _fn, args, kwargs) -> T:
        for item in _col:
            _fn(item, *args, **kwargs)
        return self

    def each_constraint(self: T, _fn=set_attrs, *args, **kwargs) -> T:
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

    def each_shape(self: T, _fn=set_attrs, *args, **kwargs) -> T:
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
        assert len(v) == 2
        v2 = lib.cpBodyLocalToWorld(self._body, v)
        return Vec2d(v2.x, v2.y)

    def world_to_local(self, v: Tuple[float, float]) -> Vec2d:
        """Convert world space coordinates to body local coordinates

        :param v: Vector in world space coordinates
        """
        assert len(v) == 2
        v2 = lib.cpBodyWorldToLocal(self._body, v)
        return Vec2d(v2.x, v2.y)

    def velocity_at_world_point(self, point: Tuple[float, float]) -> Vec2d:
        """Get the absolute velocity of the rigid body at the given world
        point

        It's often useful to know the absolute velocity of a point on the
        surface of a body since the angular velocity affects everything
        except the center of gravity.
        """
        assert len(point) == 2
        v = lib.cpBodyGetVelocityAtWorldPoint(self._body, point)
        return Vec2d(v.x, v.y)

    def velocity_at_local_point(self, point: Tuple[float, float]) -> Vec2d:
        """Get the absolute velocity of the rigid body at the given body
        local point
        """
        assert len(point) == 2
        v = lib.cpBodyGetVelocityAtLocalPoint(self._body, point)
        return Vec2d(v.x, v.y)


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
