__docformat__ = "reStructuredText"

import logging
import platform
import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    TypeVar,
)

import sidekick.api as sk

from . import _chipmunk_cffi, _version, vec2d
from ._mixins import PickleMixin, FilterElementsMixin, _State
from .body import Body
from .collision_handler import CollisionHandler
from .constraints import Constraint
from .contact_point_set import ContactPointSet
from .query_info import PointQueryInfo, SegmentQueryInfo, ShapeQueryInfo
from .shape_filter import ShapeFilter
from .shapes import Shape
from .space_debug_draw_options import SpaceDebugDrawOptions
from .util import void, init_attributes
from .vec2d import Vec2d, vec2d_from_cffi

if TYPE_CHECKING:
    from .bb import BB

cp = _chipmunk_cffi.lib
ffi = _chipmunk_cffi.ffi

_AddableObjects = Union[Body, Shape, Constraint]
T = TypeVar("T")


class Space(PickleMixin, FilterElementsMixin):
    """Spaces are the basic unit of simulation. You add rigid bodies, shapes
    and joints to it and then step them all forward together through time.

    A Space can be copied and pickled. Note that any post step callbacks are
    not copied. Also note that some internal collision cache data is not copied,
    which can make the simulation a bit unstable the first few steps of the
    fresh copy.

    Custom properties set on the space will also be copied/pickled.

    Any collision handlers will also be copied/pickled. Note that depending on
    the pickle protocol used there are some restrictions on what functions can
    be copied/pickled.

    Example::

    >>> import easymunk, pickle
    >>> space = easymunk.Space()
    >>> space2 = space.copy()
    >>> space3 = pickle.loads(pickle.dumps(space))
    """

    _pickle_attrs_init = [
        *PickleMixin._pickle_attrs_init,
        "threaded",
    ]
    _pickle_attrs_general = [
        *PickleMixin._pickle_attrs_general,
        "iterations",
        "gravity",
        "damping",
        "idle_speed_threshold",
        "sleep_time_threshold",
        "collision_slop",
        "collision_bias",
        "collision_persistence",
        "threads",
    ]
    _init_kwargs = set(_pickle_attrs_general)

    iterations: int = property(
        lambda self: cp.cpSpaceGetIterations(self._space),
        lambda self, value: void(cp.cpSpaceSetIterations(self._space, value)),
        doc="""Iterations allow you to control the accuracy of the solver.

        Defaults to 10.

        Pymunk uses an iterative solver to figure out the forces between
        objects in the space. What this means is that it builds a big list of
        all of the collisions, joints, and other constraints between the
        bodies and makes several passes over the list considering each one
        individually. The number of passes it makes is the iteration count,
        and each iteration makes the solution more accurate. If you use too
        many iterations, the physics should look nice and solid, but may use
        up too much CPU time. If you use too few iterations, the simulation
        may seem mushy or bouncy when the objects should be solid. Setting
        the number of iterations lets you balance between CPU usage and the
        accuracy of the physics. Pymunk's default of 10 iterations is
        sufficient for most simple games.
        """,
    )
    gravity: Vec2d = property(
        lambda self: vec2d_from_cffi(cp.cpSpaceGetGravity(self._space)),
        lambda self, g: void(cp.cpSpaceSetGravity(self._space, g)),
        doc="""Global gravity applied to the space.

        Defaults to (0,0). Can be overridden on a per body basis by writing
        custom integration functions and set it on the body:
        :py:meth:`pymunk.Body.velocity_func`.
        """,
    )
    damping: float = property(
        lambda self: cp.cpSpaceGetDamping(self._space),
        lambda self, damping: void(cp.cpSpaceSetDamping(self._space, damping)),
        doc="""Amount of simple damping to apply to the space.

        A value of 0.9 means that each body will lose 10% of its velocity per
        second. Defaults to 1. Like gravity, it can be overridden on a per
        body basis.
        """,
    )
    idle_speed_threshold = property(
        lambda self: cp.cpSpaceGetIdleSpeedThreshold(self._space),
        lambda self, value: void(cp.cpSpaceSetIdleSpeedThreshold(self._space, value)),
        doc="""Speed threshold for a body to be considered idle.

        The default value of 0 means the space estimates a good threshold
        based on gravity.
        """,
    )
    sleep_time_threshold = property(
        lambda self: cp.cpSpaceGetSleepTimeThreshold(self._space),
        lambda self, value: void(cp.cpSpaceSetSleepTimeThreshold(self._space, value)),
        doc="""Time a group of bodies must remain idle in order to fall
        asleep.

        The default value of `inf` disables the sleeping algorithm.
        """,
    )
    collision_slop: float = property(
        lambda self: cp.cpSpaceGetCollisionSlop(self._space),
        lambda self, value: void(cp.cpSpaceSetCollisionSlop(self._space, value)),
        doc="""Amount of overlap between shapes that is allowed.

        To improve stability, set this as high as you can without noticeable
        overlapping. It defaults to 0.1.
        """,
    )
    collision_bias = property(
        lambda self: cp.cpSpaceGetCollisionBias(self._space),
        lambda self, value: void(cp.cpSpaceSetCollisionBias(self._space, value)),
        doc="""Determines how fast overlapping shapes are pushed apart.

        Pymunk allows fast moving objects to overlap, then fixes the overlap
        over time. Overlapping objects are unavoidable even if swept
        collisions are supported, and this is an efficient and stable way to
        deal with overlapping objects. The bias value controls what
        percentage of overlap remains unfixed after a second and defaults
        to ~0.2%. Valid values are in the range from 0 to 1, but using 0 is
        not recommended for stability reasons. The default value is
        calculated as cpfpow(1.0f - 0.1f, 60.0f) meaning that pymunk attempts
        to correct 10% of error ever 1/60th of a second.

        ..Note::
            Very very few games will need to change this value.
        """,
    )
    collision_persistence: int = property(
        lambda self: cp.cpSpaceGetCollisionPersistence(self._space),
        lambda self, value: void(cp.cpSpaceSetCollisionPersistence(self._space, value)),
        doc="""The number of frames the space keeps collision solutions
        around for.

        Helps prevent jittering contacts from getting worse. This defaults
        to 3.

        ..Note::
            Very very few games will need to change this value.
        """,
    )
    current_time_step: int = property(
        lambda self: cp.cpSpaceGetCurrentTimeStep(self._space),
        doc="""Retrieves the current (if you are in a callback from
        Space.step()) or most recent (outside of a Space.step() call)
        timestep.
        """,
    )
    threads: int = property(
        lambda self: int(cp.cpHastySpaceGetThreads(self._space))
        if self.threaded
        else 1,
        lambda self, n: void(
            self.threaded and cp.cpHastySpaceSetThreads(self._space, n)
        ),
        doc="""The number of threads to use for running the step function. 
        
        Only valid when the Space was created with threaded=True. Currently the 
        max limit is 2, setting a higher value wont have any effect. The 
        default is 1 regardless if the Space was created with threaded=True, 
        to keep determinism in the simulation. Note that Windows does not 
        support the threaded solver.
        """,
    )

    @property
    def shapes(self) -> List[Shape]:
        """A list of all the shapes added to this space

        (includes both static and non-static)
        """
        return list(self._shapes.values())

    @property
    def bodies(self) -> List[Body]:
        """A list of the bodies added to this space"""
        return list(self._bodies)

    @property
    def constraints(self) -> List[Constraint]:
        """A list of the constraints added to this space"""
        return list(self._constraints)

    @sk.lazy
    def static_body(self) -> Body:
        """A dedicated static body for the space.

        You don't have to use it, but many times it can be convenient to have
        a static body together with the space.
        """
        body = Body(body_type=Body.STATIC)
        body._space = weakref.proxy(self)
        cp.cpSpaceAddBody(self._space, body._body)
        return body

    @property
    def kinetic_energy(self):
        """
        Total kinetic energy of dynamic bodies.
        """
        bodies = self.filter_bodies(body_type=Body.DYNAMIC)
        return sum(b.kinetic_energy for b in bodies)

    @property
    def gravitational_energy(self):
        """
        Potential energy of dynamic bodies due to gravity.
        """
        bodies = self.filter_bodies(body_type=Body.DYNAMIC)
        return sum(b.gravitational_energy for b in bodies)

    @property
    def potential_energy(self):
        """
        Sum of gravitational energy and all tracked sources of potential
        energies.
        """
        energy = self.gravitational_energy
        for force in self._forces:
            try:
                acc = force.potential_energy
            except AttributeError:
                pass
            else:
                energy += acc
        return energy

    @property
    def energy(self):
        """
        The sum of kinetic and potential energy.
        """
        return self.potential_energy + self.kinetic_energy

    @property
    def center_of_mass(self):
        """
        Center of mass position of all dynamic objects.
        """
        m_acc = 0.0
        pos_m_acc = Vec2d(0, 0)
        for o in self.filter_bodies(body_type=Body.DYNAMIC):
            m_acc += o.mass
            pos_m_acc += o.mass * o.local_to_world(o.center_of_mass)
        return pos_m_acc / m_acc

    @property
    def linear_momentum(self):
        """
        Total Linear momentum assigned to dynamic objects.
        """
        momentum = Vec2d(0, 0)
        for o in self.filter_bodies(body_type=Body.DYNAMIC):
            momentum += o.mass * o.velocity
        return momentum

    @property
    def angular_momentum(self):
        """
        Total angular momentum assigned to dynamic objects.
        """
        momentum = 0
        for o in self.filter_bodies(body_type=Body.DYNAMIC):
            momentum += o.moment * o.angular_velocity
            momentum += o.local_to_world(o.center_of_mass).cross(o.velocity)
        return momentum

    def __init__(self, threaded: bool = False, **kwargs) -> None:
        """Create a new instance of the Space.

        If you set threaded=True the step function will run in threaded mode
        which might give a speedup. Note that even when you set threaded=True
        you still have to set Space.threads=2 to actually use more than one
        thread.

        Also note that threaded mode is not available on Windows, and setting
        threaded=True has no effect on that platform.
        """

        self.threaded = threaded and platform.system() != "Windows"

        if self.threaded:
            cp_space = cp.cpHastySpaceNew()
            freefunc = cp.cpHastySpaceFree
        else:
            cp_space = cp.cpSpaceNew()
            freefunc = cp.cpSpaceFree
        self._space = ffi.gc(cp_space, cffi_free_space(freefunc))

        # To prevent the gc to collect the callbacks.
        self._handlers: Dict[Any, CollisionHandler] = {}
        self._post_step_callbacks: Dict[Any, Callable[["Space"], None]] = {}

        self._removed_shapes: Dict[int, Shape] = {}
        self._shapes: Dict[int, Shape] = {}
        self._bodies: Dict[Body, None] = {}
        self._constraints: Dict[Constraint, None] = {}
        self._add_later: Set[_AddableObjects] = set()
        self._remove_later: Set[_AddableObjects] = set()
        self._locked = False
        self._forces = []

        # Save attributes
        init_attributes(self, self._init_kwargs, kwargs)

    def _get_self(self) -> "Space":
        return self

    def __getstate__(self) -> _State:
        """Return the state of this object

        This method allows the usage of the :mod:`copy` and :mod:`pickle`
        modules with this class.
        """
        d = super(Space, self).__getstate__()

        d["special"].append(("easymunk_version", _version.version))
        d["special"].append(("bodies", self.bodies))
        if "static_body" in self.__dict__:
            d["special"].append(("static_body", self.static_body))

        d["special"].append(("shapes", self.shapes))
        d["special"].append(("constraints", self.constraints))

        handlers = []
        for k, v in self._handlers.items():
            h: Dict[str, Any] = {}
            if v._begin_base is not None:
                h["_begin_base"] = v._begin_base
            if v._pre_solve_base is not None:
                h["_pre_solve_base"] = v._pre_solve_base
            if v._post_solve_base is not None:
                h["_post_solve_base"] = v._post_solve_base
            if v._separate_base is not None:
                h["_separate_base"] = v._separate_base
            handlers.append((k, h))

        d["special"].append(("_handlers", handlers))

        return d

    def __setstate__(self, state: _State) -> None:
        """Unpack this object from a saved state.

        This method allows the usage of the :mod:`copy` and :mod:`pickle`
        modules with this class.
        """
        super(Space, self).__setstate__(state)

        for k, v in state["special"]:
            if k == "easymunk_version":
                assert _version.version == v, (
                    f"Pymunk version {v} of pickled object does not match current Pymunk "
                    f""
                    f""
                    f""
                    f""
                    f""
                    f"version {_version.version}"
                )
            elif k == "bodies":
                self.add(*v)
            elif k == "static_body":
                self.static_body = v
                v._space = weakref.proxy(self)
                cp.cpSpaceAddBody(self._space, v._body)
            elif k == "shapes":
                self.add(*v)
            elif k == "constraints":
                self.add(*v)
            elif k == "_handlers":
                for k2, hd in v:
                    if k2 == None:
                        h = self.add_default_collision_handler()
                    elif isinstance(k2, tuple):
                        h = self.add_collision_handler(k2[0], k2[1])
                    else:
                        h = self.add_wildcard_collision_handler(k2)
                    if "_begin_base" in hd:
                        h.begin = hd["_begin_base"]
                    if "_pre_solve_base" in hd:
                        h.pre_solve = hd["_pre_solve_base"]
                    if "_post_solve_base" in hd:
                        h.post_solve = hd["_post_solve_base"]
                    if "_separate_base" in hd:
                        h.separate = hd["_separate_base"]

    def add(self: T, *objs: _AddableObjects) -> T:
        """Add one or many shapes, bodies or constraints (joints) to the space

        Unlike Chipmunk and earlier versions of pymunk its now allowed to add
        objects even from a callback during the simulation step. However, the
        add will not be performed until the end of the step.
        """

        if self._locked:
            self._add_later.update(objs)
            return self

        # add bodies first, since the shapes require their bodies to be
        # already added. This allows code like space.add(shape, body).
        for o in objs:
            if isinstance(o, Body):
                self._add_body(o)

        for o in objs:
            if isinstance(o, Body):
                pass
            elif isinstance(o, Shape):
                self._add_shape(o)
            elif isinstance(o, Constraint):
                self._add_constraint(o)
            else:
                raise Exception(f"Unsupported type  {type(o)} of {o}.")

        return self

    def remove(self: T, *objs: _AddableObjects) -> T:
        """Remove one or many shapes, bodies or constraints from the space

        Unlike Chipmunk and earlier versions of Pymunk its now allowed to
        remove objects even from a callback during the simulation step.
        However, the removal will not be performed until the end of the step.

        .. Note::
            When removing objects from the space, make sure you remove any
            other objects that reference it. For instance, when you remove a
            body, remove the joints and shapes attached to it.
        """
        if self._locked:
            self._remove_later.update(objs)
            return self

        for o in objs:
            if isinstance(o, Body):
                self._remove_body(o)
            elif isinstance(o, Shape):
                self._remove_shape(o)
            elif isinstance(o, Constraint):
                self._remove_constraint(o)
            else:
                raise Exception(f"Unsupported type  {type(o)} of {o}.")
        return self

    def _add_shape(self, shape: "Shape") -> None:
        if shape._id in self._shapes:
            raise ValueError("shape already added to space")

        shape._space = weakref.proxy(self)
        self._shapes[shape._id] = shape
        cp.cpSpaceAddShape(self._space, shape._shape)

    def _add_body(self, body: "Body") -> None:
        if body in self._bodies:
            raise ValueError("body already added to space")

        body._space = weakref.proxy(self)
        self._bodies[body] = None
        cp.cpSpaceAddBody(self._space, body._body)

    def _add_constraint(self, constraint: "Constraint") -> None:
        if constraint in self._constraints:
            ValueError("constraint already added to space")

        self._constraints[constraint] = None
        cp.cpSpaceAddConstraint(self._space, constraint._constraint)

    def _remove_shape(self, shape: "Shape") -> None:
        if shape._id not in self._shapes:
            raise ValueError("shape not in space, already removed?")
        self._removed_shapes[shape._id] = shape

        # During GC at program exit sometimes the shape might already be removed. Then
        # skip this step.
        if cp.cpSpaceContainsShape(self._space, shape._shape):
            cp.cpSpaceRemoveShape(self._space, shape._shape)
        del self._shapes[shape._id]

    def _remove_body(self, body: "Body") -> None:
        if body not in self._bodies:
            raise ValueError("body not in space, already removed?")
        body._space = None

        # During GC at program exit sometimes the shape might already be removed. Then
        # skip this step.
        if cp.cpSpaceContainsBody(self._space, body._body):
            cp.cpSpaceRemoveBody(self._space, body._body)
        del self._bodies[body]

    def _remove_constraint(self, constraint: "Constraint") -> None:
        """Removes a constraint from the space"""
        if constraint not in self._constraints:
            raise ValueError("constraint not in space, already removed?")

        # During GC at program exit sometimes the constraint might already be removed.
        # Then skip this steip.
        if cp.cpSpaceContainsConstraint(self._space, constraint._constraint):
            cp.cpSpaceRemoveConstraint(self._space, constraint._constraint)
        del self._constraints[constraint]

    def reindex_shape(self: T, shape: Shape) -> T:
        """Update the collision detection data for a specific shape in the
        space.
        """
        cp.cpSpaceReindexShape(self._space, shape._shape)
        return self

    def reindex_shapes_for_body(self: T, body: Body) -> T:
        """Reindex all the shapes for a certain body."""
        cp.cpSpaceReindexShapesForBody(self._space, body._body)
        return self

    def reindex_static(self: T) -> T:
        """Update the collision detection info for the static shapes in the
        space. You only need to call this if you move one of the static shapes.
        """
        cp.cpSpaceReindexStatic(self._space)
        return self

    def use_spatial_hash(self: T, dim: float, count: int) -> T:
        """Switch the space to use a spatial hash instead of the bounding box
        tree.

        Pymunk supports two spatial indexes. The default is an axis-aligned
        bounding box tree inspired by the one used in the Bullet Physics
        library, but caching of overlapping leaves was added to give it very
        good temporal coherence. The tree requires no tuning, and most games
        will find that they get the best performance using from the tree. The
        other available spatial index type available is a spatial hash, which
        can be much faster when you have a very large number (1000s) of
        objects that are all the same size. For smaller numbers of objects,
        or objects that vary a lot in size, the spatial hash is usually much
        slower. It also requires tuning (usually through experimentation) to
        get the best possible performance.

        The spatial hash data is fairly size sensitive. dim is the size of
        the hash cells. Setting dim to the average collision shape size is
        likely to give the best performance. Setting dim too small will cause
        the shape to be inserted into many cells, setting it too low will
        cause too many objects into the same hash slot.

        count is the suggested minimum number of cells in the hash table. If
        there are too few cells, the spatial hash will return many false
        positives. Too many cells will be hard on the cache and waste memory.
        Setting count to ~10x the number of objects in the space is probably a
        good starting point. Tune from there if necessary.

        :param dim: the size of the hash cells
        :param count: the suggested minimum number of cells in the hash table
        """
        cp.cpSpaceUseSpatialHash(self._space, dim, count)
        return self

    def step(self, dt: float) -> None:
        """Update the space for the given time step.

        Using a fixed time step is highly recommended. Doing so will increase
        the efficiency of the contact persistence, requiring an order of
        magnitude fewer iterations to resolve the collisions in the usual case.

        It is not the same to call step 10 times with a dt of 0.1 and
        calling it 100 times with a dt of 0.01 even if the end result is
        that the simulation moved forward 100 units. Performing  multiple
        calls with a smaller dt creates a more stable and accurate
        simulation. Therefor it sometimes make sense to have a little for loop
        around the step call, like in this example:

        >>> import easymunk
        >>> s = easymunk.Space()
        >>> steps = 10
        >>> for x in range(steps): # move simulation forward 0.1 seconds:
        ...     s.step(0.1 / steps)

        :param dt: Time step length
        """
        try:
            self._locked = True
            if self.threaded:
                cp.cpHastySpaceStep(self._space, dt)
            else:
                cp.cpSpaceStep(self._space, dt)
            self._removed_shapes = {}
        finally:
            self._locked = False

        self.add(*self._add_later)
        self._add_later.clear()
        for obj in self._remove_later:
            self.remove(obj)
        self._remove_later.clear()

        for key in self._post_step_callbacks:
            self._post_step_callbacks[key](self)

        self._post_step_callbacks = {}

    def add_collision_handler(
        self, collision_type_a: int, collision_type_b: int
    ) -> CollisionHandler:
        """Return the :py:class:`CollisionHandler` for collisions between
        objects of type collision_type_a and collision_type_b.

        Fill the desired collision callback functions, for details see the
        :py:class:`CollisionHandler` object.

        Whenever shapes with collision types (:py:attr:`Shape.collision_type`)
        a and b collide, this handler will be used to process the collision
        events. When a new collision handler is created, the callbacks will all be
        set to builtin callbacks that perform the default behavior (call the
        wildcard handlers, and accept all collisions).

        :param int collision_type_a: Collision type a
        :param int collision_type_b: Collision type b

        :rtype: :py:class:`CollisionHandler`
        """
        key = min(collision_type_a, collision_type_b), max(
            collision_type_a, collision_type_b
        )
        if key in self._handlers:
            return self._handlers[key]

        h = cp.cpSpaceAddCollisionHandler(
            self._space, collision_type_a, collision_type_b
        )
        ch = CollisionHandler(h, self)
        self._handlers[key] = ch
        return ch

    def add_wildcard_collision_handler(self, collision_type_a: int) -> CollisionHandler:
        """Add a wildcard collision handler for given collision type.

        This handler will be used any time an object with this type collides
        with another object, regardless of its type. A good example is a
        projectile that should be destroyed the first time it hits anything.
        There may be a specific collision handler and two wildcard handlers.
        It's up to the specific handler to decide if and when to call the
        wildcard handlers and what to do with their return values.

        When a new wildcard handler is created, the callbacks will all be
        set to builtin callbacks that perform the default behavior. (accept
        all collisions in :py:func:`~CollisionHandler.begin` and
        :py:func:`~CollisionHandler.pre_solve`, or do nothing for
        :py:func:`~CollisionHandler.post_solve` and
        :py:func:`~CollisionHandler.separate`.

        :param int collision_type_a: Collision type
        :rtype: :py:class:`CollisionHandler`
        """

        if collision_type_a in self._handlers:
            return self._handlers[collision_type_a]

        h = cp.cpSpaceAddWildcardHandler(self._space, collision_type_a)
        ch = CollisionHandler(h, self)
        self._handlers[collision_type_a] = ch
        return ch

    def add_default_collision_handler(self) -> CollisionHandler:
        """Return a reference to the default collision handler or that is
        used to process all collisions that don't have a more specific
        handler.

        The default behavior for each of the callbacks is to call
        the wildcard handlers, ANDing their return values together if
        applicable.
        """
        if None in self._handlers:
            return self._handlers[None]

        _h = cp.cpSpaceAddDefaultCollisionHandler(self._space)
        h = CollisionHandler(_h, self)
        self._handlers[None] = h
        return h

    def add_post_step_callback(
        self,
        callback_function: Callable[
            ..., None
        ],  # TODO: Fix me once PEP-612 is implemented
        key: Hashable,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Add a function to be called last in the next simulation step.

        Post step callbacks are registered as a function and an object used as
        a key. You can only register one post step callback per object.

        This function was more useful with earlier versions of pymunk where
        you weren't allowed to use the add and remove methods on the space
        during a simulation step. But this function is still available for
        other uses and to keep backwards compatibility.

        .. Note::
            If you remove a shape from the callback it will trigger the
            collision handler for the 'separate' event if it the shape was
            touching when removed.

        .. Note::
            Post step callbacks are not included in pickle / copy of the space.

        :param callback_function: The callback function
        :type callback_function: `func(space : Space, key, *args, **kwargs)`
        :param Any key:
            This object is used as a key, you can only have one callback
            for a single object. It is passed on to the callback function.
        :param args: Optional parameters passed to the callback
        :param kwargs: Optional keyword parameters passed on to the callback

        :return: True if key was not previously added, False otherwise
        """

        if key in self._post_step_callbacks:
            return False

        def f(x):  # type: ignore
            callback_function(self, key, *args, **kwargs)

        self._post_step_callbacks[key] = f
        return True

    def point_query(
        self, point: Tuple[float, float], max_distance: float, shape_filter: ShapeFilter
    ) -> List[PointQueryInfo]:
        """Query space at point for shapes within the given distance range.

        The filter is applied to the query and follows the same rules as the
        collision detection. If a maxDistance of 0.0 is used, the point must
        lie inside a shape. Negative max_distance is also allowed meaning that
        the point must be a under a certain depth within a shape to be
        considered a match.

        See :py:class:`ShapeFilter` for details about how the shape_filter
        parameter can be used.

        .. Note::
            Sensor shapes are included in the result (In
            :py:meth:`Space.point_query_nearest` they are not)

        :param point: Where to check for collision in the Space
        :type point: :py:class:`~vec2d.Vec2d` or (float,float)
        :param float max_distance: Match only within this distance
        :param ShapeFilter shape_filter: Only pick shapes matching the filter

        :rtype: [:py:class:`PointQueryInfo`]
        """
        assert len(point) == 2
        query_hits: List[PointQueryInfo] = []

        @ffi.callback("cpSpacePointQueryFunc")
        def cf(_shape, point, distance, gradient, data):  # type: ignore
            # space = ffi.from_handle(data)
            shape = self._get_shape(_shape)
            p = PointQueryInfo(
                shape, Vec2d(point.x, point.y), distance, Vec2d(gradient.x, gradient.y)
            )
            nonlocal query_hits
            query_hits.append(p)

        data = ffi.new_handle(self)
        cp.cpSpacePointQuery(self._space, point, max_distance, shape_filter, cf, data)
        return query_hits

    def _get_shape(self, _shape: Any) -> Optional[Shape]:
        if not bool(_shape):
            return None

        shapeid = int(ffi.cast("int", cp.cpShapeGetUserData(_shape)))

        if shapeid in self._shapes:
            return self._shapes[shapeid]
        elif shapeid in self._removed_shapes:
            return self._removed_shapes[shapeid]
        else:
            return None

    def point_query_nearest(
        self, point: Tuple[float, float], max_distance: float, shape_filter: ShapeFilter
    ) -> Optional[PointQueryInfo]:
        """Query space at point the nearest shape within the given distance
        range.

        The filter is applied to the query and follows the same rules as the
        collision detection. If a maxDistance of 0.0 is used, the point must
        lie inside a shape. Negative max_distance is also allowed meaning that
        the point must be a under a certain depth within a shape to be
        considered a match.

        See :py:class:`ShapeFilter` for details about how the shape_filter
        parameter can be used.

        .. Note::
            Sensor shapes are not included in the result (In
            :py:meth:`Space.point_query` they are)

        :param point: Where to check for collision in the Space
        :type point: :py:class:`~vec2d.Vec2d` or (float,float)
        :param float max_distance: Match only within this distance
        :param ShapeFilter shape_filter: Only pick shapes matching the filter

        :rtype: :py:class:`PointQueryInfo` or None
        """
        assert len(point) == 2
        info = ffi.new("cpPointQueryInfo *")
        _shape = cp.cpSpacePointQueryNearest(
            self._space, point, max_distance, shape_filter, info
        )

        shape = self._get_shape(_shape)

        if shape != None:
            return PointQueryInfo(
                shape,
                Vec2d(info.point.x, info.point.y),
                info.distance,
                Vec2d(info.gradient.x, info.gradient.y),
            )
        return None

    def segment_query(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        radius: float,
        shape_filter: ShapeFilter,
    ) -> List[SegmentQueryInfo]:
        """Query space along the line segment from start to end with the
        given radius.

        The filter is applied to the query and follows the same rules as the
        collision detection.

        See :py:class:`ShapeFilter` for details about how the shape_filter
        parameter can be used.

        .. Note::
            Sensor shapes are included in the result (In
            :py:meth:`Space.segment_query_first` they are not)

        :param start: Starting point
        :param end: End point
        :param float radius: Radius
        :param ShapeFilter shape_filter: Shape filter

        :rtype: [:py:class:`SegmentQueryInfo`]
        """
        assert len(start) == 2
        assert len(end) == 2
        query_hits: List[SegmentQueryInfo] = []

        @ffi.callback("cpSpaceSegmentQueryFunc")
        def cf(_shape, point, normal, alpha, data):  # type: ignore
            shape = self._get_shape(_shape)
            p = SegmentQueryInfo(
                shape, Vec2d(point.x, point.y), Vec2d(normal.x, normal.y), alpha
            )
            query_hits.append(p)

        data = ffi.new_handle(self)
        cp.cpSpaceSegmentQuery(self._space, start, end, radius, shape_filter, cf, data)
        return query_hits

    def segment_query_first(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        radius: float,
        shape_filter: ShapeFilter,
    ) -> Optional[SegmentQueryInfo]:
        """Query space along the line segment from start to end with the
        given radius.

        The filter is applied to the query and follows the same rules as the
        collision detection.

        .. Note::
            Sensor shapes are not included in the result (In
            :py:meth:`Space.segment_query` they are)

        See :py:class:`ShapeFilter` for details about how the shape_filter
        parameter can be used.

        :rtype: :py:class:`SegmentQueryInfo` or None
        """
        assert len(start) == 2
        assert len(end) == 2
        info = ffi.new("cpSegmentQueryInfo *")
        _shape = cp.cpSpaceSegmentQueryFirst(
            self._space, start, end, radius, shape_filter, info
        )

        shape = self._get_shape(_shape)
        if shape != None:
            return SegmentQueryInfo(
                shape,
                Vec2d(info.point.x, info.point.y),
                Vec2d(info.normal.x, info.normal.y),
                info.alpha,
            )
        return None

    def bb_query(self, bb: "BB", shape_filter: ShapeFilter) -> List[Shape]:
        """Query space to find all shapes near bb.

        The filter is applied to the query and follows the same rules as the
        collision detection.

        .. Note::
            Sensor shapes are included in the result

        :param bb: Bounding box
        :param shape_filter: Shape filter

        :rtype: [:py:class:`Shape`]
        """

        query_hits = []

        @ffi.callback("cpSpaceBBQueryFunc")
        def cf(_shape, data):  # type: ignore
            shape = self._get_shape(_shape)
            assert shape is not None
            nonlocal query_hits
            query_hits.append(shape)

        data = ffi.new_handle(self)
        cp.cpSpaceBBQuery(self._space, bb, shape_filter, cf, data)
        return query_hits

    def shape_query(self, shape: Shape) -> List[ShapeQueryInfo]:
        """Query a space for any shapes overlapping the given shape

        .. Note::
            Sensor shapes are included in the result

        :param shape: Shape to query with
        :type shape: :py:class:`Circle`, :py:class:`Poly` or :py:class:`Segment`

        :rtype: [:py:class:`ShapeQueryInfo`]
        """

        query_hits = []

        @ffi.callback("cpSpaceShapeQueryFunc")
        def cf(_shape, _points, _data):  # type: ignore
            found_shape = self._get_shape(_shape)
            point_set = ContactPointSet._from_cp(_points)
            info = ShapeQueryInfo(found_shape, point_set)
            nonlocal query_hits
            query_hits.append(info)

        data = ffi.new_handle(self)
        cp.cpSpaceShapeQuery(self._space, shape._shape, cf, data)

        return query_hits

    def debug_draw(self, options: SpaceDebugDrawOptions) -> None:
        """Debug draw the current state of the space using the supplied drawing
        options.

        If you use a graphics backend that is already supported, such as pygame
        and pyglet, you can use the predefined options in their x_util modules,
        for example :py:class:`pygame_util.DrawOptions`.

        Its also possible to write your own graphics backend, see
        :py:class:`SpaceDebugDrawOptions`.

        If you require any advanced or optimized drawing its probably best to
        not use this function for the drawing since its meant for debugging
        and quick scripting.

        :type options: :py:class:`SpaceDebugDrawOptions`
        """
        if options._use_chipmunk_debug_draw:
            h = ffi.new_handle(self)
            # we need to hold h until the end of cpSpaceDebugDraw to prevent GC
            options._options.data = h

            with options:
                cp.cpSpaceDebugDraw(self._space, options._options)
        else:
            for shape in self.shapes:
                options.draw_shape(shape)


@sk.curry(2)
def cffi_free_space(free_cb, cp_space):
    logging.debug("spacefree start %s", cp_space)
    cp_shapes = []
    cp_constraints = []
    cp_bodies = []

    @ffi.callback("cpSpaceShapeIteratorFunc")
    def cf1(shape, _):
        cp_shapes.append(shape)

    @ffi.callback("cpSpaceConstraintIteratorFunc")
    def cf2(constraint, _):
        cp_constraints.append(constraint)

    @ffi.callback("cpSpaceBodyIteratorFunc")
    def cf3(body, _):
        cp_bodies.append(body)

    cp.cpSpaceEachShape(cp_space, cf1, ffi.NULL)
    for cp_shape in cp_shapes:
        logging.debug("free %s %s", cp_space, cp_shape)
        cp.cpSpaceRemoveShape(cp_space, cp_shape)
        cp.cpShapeSetBody(cp_shape, ffi.NULL)

    cp.cpSpaceEachConstraint(cp_space, cf2, ffi.NULL)
    for cp_constraint in cp_constraints:
        logging.debug("free %s %s", cp_space, cp_constraint)
        cp.cpSpaceRemoveConstraint(cp_space, cp_constraint)

    cp.cpSpaceEachBody(cp_space, cf3, ffi.NULL)
    for cp_body in cp_bodies:
        logging.debug("free %s %s", cp_space, cp_body)
        cp.cpSpaceRemoveBody(cp_space, cp_body)

    logging.debug("spacefree free %s", cp_space)
    free_cb(cp_space)
