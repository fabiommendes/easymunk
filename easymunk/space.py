__docformat__ = "reStructuredText"

import logging
import platform
import weakref
from functools import lru_cache
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
    Iterator,
)

import sidekick.api as sk

from . import _chipmunk_cffi, _version
from ._mixins import PickleMixin, FilterElementsMixin, _State
from .arbiter import Arbiter
from .body import Body
from .collections import Shapes, Bodies, Constraints
from .collision_handler import CollisionHandler
from .constraints import Constraint
from .contact_point_set import contact_point_set_from_cffi
from .query_info import PointQueryInfo, SegmentQueryInfo, ShapeQueryInfo
from .shape_filter import ShapeFilter
from .shapes import Shape
from .util import (
    void,
    init_attributes,
    shape_id,
    cffi_body,
    get_nursery,
    get_cffi_ref,
    clear_nursery,
)
from .vec2d import Vec2d, vec2d_from_cffi, VecLike

if TYPE_CHECKING:
    from .bb import BB
    from .space_debug_draw_options import SpaceDebugDrawOptions

cp = _chipmunk_cffi.lib
ffi = _chipmunk_cffi.ffi

DEBUG_DRAW_PYGAME = sk.import_later(".pygame:DrawOptions", package=__package__)
DEBUG_DRAW_PYXEL = sk.import_later(".pyxel:DrawOptions", package=__package__)
DEBUG_DRAW_PYGLET = sk.import_later(".pyglet:DrawOptions", package=__package__)

_AddableObjects = Union[Body, Shape, Constraint]
S = TypeVar("S", bound="Space")

POINT_QUERY_ARGS = """
        Args:
            point:
                Where to check for collision in the Space.
            distance:
                Match within this tolerance. If a distance of 0.0 is used, the
                point must lie inside a shape. Negative values mean that
                the point must be a under a certain depth within a shape to be
                considered a match.
            filter:
                Only pick shapes matching the filter.
"""


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

    iterations: int
    iterations = property(  # type: ignore
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
    gravity: Vec2d
    gravity = property(  # type: ignore
        lambda self: vec2d_from_cffi(cp.cpSpaceGetGravity(self._space)),
        lambda self, g: void(cp.cpSpaceSetGravity(self._space, g)),
        doc="""Global gravity applied to the space.

        Defaults to (0,0). Can be overridden on a per body basis by writing
        custom integration functions and set it on the body:
        :py:meth:`easymunk.Body.velocity_func`.
        """,
    )
    damping: float
    damping = property(  # type: ignore
        lambda self: cp.cpSpaceGetDamping(self._space),
        lambda self, damping: void(cp.cpSpaceSetDamping(self._space, damping)),
        doc="""Amount of simple damping to apply to the space.

        A value of 0.9 means that each body will lose 10% of its velocity per
        second. Defaults to 1. Like gravity, it can be overridden on a per
        body basis.
        """,
    )
    idle_speed_threshold: float
    idle_speed_threshold = property(  # type: ignore
        lambda self: cp.cpSpaceGetIdleSpeedThreshold(self._space),
        lambda self, value: void(cp.cpSpaceSetIdleSpeedThreshold(self._space, value)),
        doc="""Speed threshold for a body to be considered idle.

        The default value of 0 means the space estimates a good threshold
        based on gravity.
        """,
    )
    sleep_time_threshold: float
    sleep_time_threshold = property(  # type: ignore
        lambda self: cp.cpSpaceGetSleepTimeThreshold(self._space),
        lambda self, value: void(cp.cpSpaceSetSleepTimeThreshold(self._space, value)),
        doc="""Time a group of bodies must remain idle in order to fall
        asleep.

        The default value of `inf` disables the sleeping algorithm.
        """,
    )
    collision_slop: float
    collision_slop = property(  # type: ignore
        lambda self: cp.cpSpaceGetCollisionSlop(self._space),
        lambda self, value: void(cp.cpSpaceSetCollisionSlop(self._space, value)),
        doc="""Amount of overlap between shapes that is allowed.

        To improve stability, set this as high as you can without noticeable
        overlapping. It defaults to 0.1.
        """,
    )
    collision_bias: float
    collision_bias = property(  # type: ignore
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
    collision_persistence: int
    collision_persistence = property(  # type: ignore
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
    current_time_step: int
    current_time_step = property(  # type: ignore
        lambda self: cp.cpSpaceGetCurrentTimeStep(self._space),
        doc="""Retrieves the current (if you are in a callback from
        Space.step()) or most recent (outside of a Space.step() call)
        timestep.
        """,
    )
    threads: int
    threads = property(  # type: ignore
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

    @sk.lazy
    def shapes(self) -> Shapes:
        """A list of all the shapes added to this space

        (includes both static and non-static)
        """
        return Shapes(self, self._shapes.values())

    @sk.lazy
    def bodies(self) -> Bodies:
        """A list of the bodies added to this space"""
        return Bodies(self, self._bodies)

    @sk.lazy
    def constraints(self) -> Constraints:
        """A list of the constraints added to this space"""
        return Constraints(self, self._constraints)

    @sk.lazy
    def static_body(self) -> Body:
        """A dedicated static body for the space.

        You don't have to use it, but many times it can be convenient to have
        a static body together with the space.
        """
        body = Body(body_type=Body.STATIC)
        body._space = weakref.proxy(self)

        cp.cpSpaceAddBody(self._space, cffi_body(body))
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
        for force in self._forces:  # TODO: implement external forces
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
        self._space: Any = ffi.gc(cp_space, cffi_free_space(freefunc))

        # To prevent the gc to collect the callbacks.
        self._handlers: Dict[Any, CollisionHandler] = {}
        self._post_step_callbacks: Dict[Any, Callable[["Space"], None]] = {}

        self._removed_shapes: Dict[int, Shape] = {}
        self._shapes: Dict[int, Shape] = {}
        self._bodies: Set[Body] = set()
        self._constraints: Set[Constraint] = set()
        self._add_later: Set[_AddableObjects] = set()
        self._remove_later: Set[_AddableObjects] = set()
        self._locked: bool = False

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
                if _version.version != v:
                    raise ValueError(
                        f"Pymunk version {v} of pickled object does not match current "
                        f"Pymunk version {_version.version}"
                    )
            elif k == "bodies":
                self.add(*v)
            elif k == "static_body":
                self.static_body = v
                v._space = weakref.proxy(self)
                cp.cpSpaceAddBody(self._space, v._cffi_ref)
            elif k == "shapes":
                self.add(*v)
            elif k == "constraints":
                self.add(*v)
            elif k == "_handlers":
                for k2, hd in v:
                    if k2 is None:
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

    def _iter_bodies(self) -> Iterator["Body"]:
        return iter(self._bodies)

    def _iter_shapes(self) -> Iterator["Shape"]:
        return iter(self._shapes.values())

    def _iter_constraints(self) -> Iterator["Constraint"]:
        return iter(self._constraints)

    def add(self: S, *objs: _AddableObjects, add_children=True) -> S:
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
        nursery = set()
        other_objs = []
        for o in objs:
            if isinstance(o, Body):
                if add_children:
                    nursery.update(get_nursery(o))
                    nursery.update(o.shapes)
                    nursery.update(o.constraints)
                self._add_body(o)
            else:
                other_objs.append(o)

        other_objs = [o for o in other_objs if o not in nursery]
        for group in (nursery, other_objs):
            for o in group:
                if isinstance(o, Shape):
                    self._add_shape(o)
                elif isinstance(o, Constraint):
                    self._add_constraint(o)
                else:
                    raise Exception(f"Unsupported type  {type(o)} of {o}.")

        return self

    def remove(self: S, *objs: _AddableObjects, remove_children=True) -> S:
        """Remove one or many shapes, bodies or constraints from the space

        If called from callback during update step, the removal will not be
        performed until the end of the step.

        .. Note::
            When removing objects from the space, make sure you remove any
            other objects that reference it. For instance, when you remove a
            body, remove the joints and shapes attached to it.
        """
        return self._remove_or_discard(objs, remove_children, False)

    def discard(self, *objs: _AddableObjects, remove_children=True):
        """
        Discard objects from space.

        Similar to remove, but do not throw errors if element is not present
        in space.
        """
        return self._remove_or_discard(objs, remove_children, True)

    def _remove_or_discard(self, objs, remove_children, discard):
        if self._locked:
            self._remove_later.update(objs)
            return self

        for o in objs:
            if isinstance(o, Body):
                self._remove_body(o, discard)
            elif isinstance(o, Shape):
                self._remove_shape(o, discard)
            elif isinstance(o, Constraint):
                self._remove_constraint(o, discard)
            else:
                raise Exception(f"Unsupported type  {type(o)} of {o}.")

        if not remove_children:
            return self

        for o in objs:
            if not isinstance(o, Body):
                continue
            for s in o.shapes:
                if s.space is self:
                    self._remove_shape(s, True)
            for c in o.constraints:
                if c.space is self:
                    self._remove_constraint(c, True)
        return self

    def _add_shape(self, shape: "Shape") -> None:
        if shape_id(shape) in self._shapes:
            return

        shape._space = weakref.proxy(self)
        self._shapes[shape_id(shape)] = shape
        cp.cpSpaceAddShape(self._space, get_cffi_ref(shape))
        clear_nursery(shape)

    def _add_body(self, body: "Body") -> None:
        if body in self._bodies:
            return

        body._space = weakref.proxy(self)
        self._bodies.add(body)
        cp.cpSpaceAddBody(self._space, get_cffi_ref(body))
        clear_nursery(body)

    def _add_constraint(self, constraint: "Constraint") -> None:
        if constraint in self._constraints:
            return

        self._constraints.add(constraint)
        cp.cpSpaceAddConstraint(self._space, get_cffi_ref(constraint))
        clear_nursery(constraint)

    def _remove_shape(self, shape: "Shape", discard: bool) -> None:
        id_ = shape_id(shape)
        if id_ not in self._shapes:
            if discard:
                return
            raise ValueError("shape not in space, already removed?")
        self._removed_shapes[id_] = shape

        # During GC at program exit sometimes the shape might already be removed. Then
        # skip this step.
        ref = get_cffi_ref(shape)
        if cp.cpSpaceContainsShape(self._space, ref):
            cp.cpSpaceRemoveShape(self._space, ref)
        del self._shapes[id_]

    def _remove_body(self, body: "Body", discard: bool) -> None:
        if body not in self._bodies:
            if discard:
                return
            raise ValueError("body not in space, already removed?")
        body._space = None

        # During GC at program exit sometimes the shape might already be removed. Then
        # skip this step.
        ref = get_cffi_ref(body)
        if cp.cpSpaceContainsBody(self._space, ref):
            cp.cpSpaceRemoveBody(self._space, ref)
        self._bodies.remove(body)

    def _remove_constraint(self, constraint: "Constraint", discard: bool) -> None:
        if constraint not in self._constraints:
            if discard:
                return
            raise ValueError("constraint not in space, already removed?")

        # During GC at program exit sometimes the constraint might already be removed.
        # Then skip this step.
        ref = get_cffi_ref(constraint)
        if cp.cpSpaceContainsConstraint(self._space, ref):
            cp.cpSpaceRemoveConstraint(self._space, ref)
        self._constraints.remove(constraint)

    def reindex_shape(self: S, shape: Shape) -> S:
        """Update the collision detection data for a specific shape in the
        space.
        """
        cp.cpSpaceReindexShape(self._space, get_cffi_ref(shape))
        return self

    def reindex_shapes_for_body(self: S, body: Body) -> S:
        """Reindex all the shapes for a certain body."""
        cp.cpSpaceReindexShapesForBody(self._space, get_cffi_ref(body))
        return self

    def reindex_static(self: S) -> S:
        """Update the collision detection info for the static shapes in the
        space. You only need to call this if you move one of the static shapes.
        """
        cp.cpSpaceReindexStatic(self._space)
        return self

    def use_spatial_hash(self: S, dim: float, count: int) -> S:
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

    def step(self: S, dt: float) -> S:
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
        self.discard(*self._remove_later)
        self._remove_later.clear()

        for key in self._post_step_callbacks:
            self._post_step_callbacks[key](self)

        self._post_step_callbacks = {}
        return self

    def add_collision_handler(self, a: int, b: int) -> CollisionHandler:
        """Return the :py:class:`CollisionHandler` for collisions between
        objects of type "a" and "b".

        Fill the desired collision callback functions, for details see the
        :py:class:`CollisionHandler` object.

        Whenever shapes with collision types (:py:attr:`Shape.collision_type`)
        a and b collide, this handler will be used to process the collision
        events. When a new collision handler is created, the callbacks will all be
        set to builtin callbacks that perform the default behavior (call the
        wildcard handlers, and accept all collisions).

        :param int a: Collision type a
        :param int b: Collision type b

        :rtype: :py:class:`CollisionHandler`
        """
        key = min(a, b), max(a, b)
        if key in self._handlers:
            return self._handlers[key]

        ptr = cp.cpSpaceAddCollisionHandler(self._space, a, b)
        self._handlers[key] = handler = CollisionHandler(ptr, self)
        return handler

    def add_wildcard_collision_handler(self, col_type: int) -> CollisionHandler:
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

        :param int col_type: Collision type
        :rtype: :py:class:`CollisionHandler`
        """

        if col_type in self._handlers:
            return self._handlers[col_type]

        ptr = cp.cpSpaceAddWildcardHandler(self._space, col_type)
        self._handlers[col_type] = handler = CollisionHandler(ptr, self)
        return handler

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

        ptr = cp.cpSpaceAddDefaultCollisionHandler(self._space)
        self._handlers[None] = handler = CollisionHandler(ptr, self)
        return handler

    def add_post_step_callback(
            self,
            callback_function: Callable[..., None],
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

        def f(_):  # type: ignore
            callback_function(self, key, *args, **kwargs)

        self._post_step_callbacks[key] = f
        return True

    # noinspection PyShadowingBuiltins
    def point_query(
            self, point: VecLike, distance: float = 0, filter: ShapeFilter = None
    ) -> List[PointQueryInfo]:
        f"""Query space at point for shapes within the given distance range.

        The filter is applied to the query and follows the same rules as the
        collision detection. See :py:class:`ShapeFilter` for details about how
        the shape_filter parameter can be used.

        {POINT_QUERY_ARGS}

        Note:
            Sensor shapes are included in the result (In
            :py:meth:`Space.point_query_nearest` they are not)

        Result:
            A list of point queries.
        """

        @ffi.callback("cpSpacePointQueryFunc")
        def cb(shape, pt, dist, gradient, _data):
            shape = shape_from_cffi(self, shape)
            if shape:
                vec = Vec2d(pt.x, pt.y)
                grad = Vec2d(gradient.x, gradient.y)
                result.append(PointQueryInfo(shape, vec, dist, grad))

        result: List[PointQueryInfo] = []
        filter_ = filter or ShapeFilter()
        data = ffi.new_handle(self)
        cp.cpSpacePointQuery(self._space, point, distance, filter_, cb, data)
        return result

    # noinspection PyShadowingBuiltins
    def point_query_nearest(
            self, point: VecLike, distance: float = 0.0, filter: ShapeFilter = None
    ) -> Optional[PointQueryInfo]:
        f"""Query space at point the nearest shape within the given distance
        range.

        {POINT_QUERY_ARGS}

        See :py:class:`ShapeFilter` for details about how the shape_filter
        parameter can be used.

        .. Note::
            Sensor shapes are not included in the result (In
            :py:meth:`Space.point_query` they are)

        Result:
            The resulting point query.
        """
        info = ffi.new("cpPointQueryInfo *")
        filter = filter or ShapeFilter()
        ptr = cp.cpSpacePointQueryNearest(self._space, point, distance, filter, info)
        shape = shape_from_cffi(self, ptr)
        if shape:
            pos = Vec2d(info.point.x, info.point.y)
            grad = Vec2d(info.gradient.x, info.gradient.y)
            return PointQueryInfo(shape, pos, info.distance, grad)
        return None

    # noinspection PyShadowingBuiltins
    def segment_query(
            self,
            start: VecLike,
            end: VecLike,
            radius: float = 0.0,
            filter: ShapeFilter = None,
    ) -> List[SegmentQueryInfo]:
        """Query space along the line segment from start to end with the
        given radius.

        The filter is applied to the query and follows the same rules as the
        collision detection.

        Args:
            start: Starting point
            end: End point
            radius: Radius
            filter: Shape filter

        Note:
            Sensor shapes are included in the result (In
            :py:meth:`Space.segment_query_first` they are not)
        """

        @ffi.callback("cpSpaceSegmentQueryFunc")
        def cb(ptr, point, normal, alpha, _data):
            shape = shape_from_cffi(self, ptr)
            if shape:
                pt = Vec2d(point.x, point.y)
                normal = Vec2d(normal.x, normal.y)
                query_hits.append(SegmentQueryInfo(shape, pt, normal, alpha))

        query_hits: List[SegmentQueryInfo] = []
        filter = filter or ShapeFilter()
        data = ffi.new_handle(self)
        cp.cpSpaceSegmentQuery(self._space, start, end, radius, filter, cb, data)
        return query_hits

    # noinspection PyShadowingBuiltins
    def segment_query_first(
            self,
            start: Tuple[float, float],
            end: Tuple[float, float],
            radius: float = 0.0,
            filter: ShapeFilter = None,
    ) -> Optional[SegmentQueryInfo]:
        """Query space along the line segment from start to end with the
        given radius.

        Similar to :py:meth:`Space.segment_query`, but return the first query
        or None.
        """

        info = ffi.new("cpSegmentQueryInfo *")
        filter = filter or ShapeFilter()
        ptr = cp.cpSpaceSegmentQueryFirst(self._space, start, end, radius, filter, info)
        shape = shape_from_cffi(self, ptr)
        if shape is not None:
            pos = Vec2d(info.point.x, info.point.y)
            normal = Vec2d(info.normal.x, info.normal.y)
            return SegmentQueryInfo(shape, pos, normal, info.alpha)
        return None

    # noinspection PyShadowingBuiltins
    def bb_query(self, bb: "BB", filter: ShapeFilter = None) -> List[Shape]:
        """Query space to find all shapes near bb.

        The filter is applied to the query and follows the same rules as the
        collision detection.

        Note:
            Sensor shapes are included in the result
        """

        @ffi.callback("cpSpaceBBQueryFunc")
        def cb(ptr, _):
            shape = shape_from_cffi(self, ptr)
            if shape:
                query_hits.append(shape)

        query_hits: List[Shape] = []
        data = ffi.new_handle(self)
        cp.cpSpaceBBQuery(self._space, bb, filter, cb, data)
        return query_hits

    def shape_query(self, shape: Shape) -> List[ShapeQueryInfo]:
        """Query a space for any shapes overlapping the given shape

        Note:
            Sensor shapes are included in the result
        """

        @ffi.callback("cpSpaceShapeQueryFunc")
        def cb(ptr, points, _):
            obj = shape_from_cffi(self, ptr)
            if obj:
                point_set = contact_point_set_from_cffi(points)
                query_hits.append(ShapeQueryInfo(obj, point_set))

        query_hits: List[ShapeQueryInfo] = []
        data = ffi.new_handle(self)
        cp.cpSpaceShapeQuery(self._space, get_cffi_ref(shape), cb, data)
        return query_hits

    def debug_draw(
            self: S, options: Union["SpaceDebugDrawOptions", str, None] = None
    ) -> S:
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

        if options is None or isinstance(options, str):
            options = get_debug_options(options)

        if options.bypass_chipmunk:
            for shape in self.shapes:
                options.draw_shape(shape)
        else:
            # We need to hold h until the end of cpSpaceDebugDraw to prevent GC
            cffi = get_cffi_ref(options)
            cffi.data = ptr = ffi.new_handle(self)
            with options:
                cp.cpSpaceDebugDraw(self._space, cffi)
            del ptr
        return self


@lru_cache
def get_debug_options(opt) -> "SpaceDebugDrawOptions":
    if opt is None:
        return get_debug_options("pygame")
    elif opt == "pygame":
        return DEBUG_DRAW_PYGAME()
    elif opt == "pyglet":
        return DEBUG_DRAW_PYGLET()
    elif opt == "pyxel":
        return DEBUG_DRAW_PYXEL()
    raise ValueError(f"invalid debug draw option: {opt}")


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


# noinspection PyProtectedMember
def shape_from_cffi(space: Space, ptr) -> Optional[Shape]:
    """Internal function that returns shape from cffi pointer."""
    if not bool(ptr):
        return None

    id_ = int(ffi.cast("int", cp.cpShapeGetUserData(ptr)))
    if id_ in space._shapes:
        return space._shapes[id_]
    elif id_ in space._removed_shapes:
        return space._removed_shapes[id_]
    else:
        return None


Arbiter._shape_from_cffi = staticmethod(shape_from_cffi)
