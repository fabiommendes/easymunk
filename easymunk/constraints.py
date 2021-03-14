# ----------------------------------------------------------------------------
# pymunk
# Copyright (c) 2007-2017 Victor Blomqvist, 2021 Fábio Macêdo Mendes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

"""A constraint is something that describes how two bodies interact with
each other. (how they constrain each other). Constraints can be simple
joints that allow bodies to pivot around each other like the bones in your
body, or they can be more abstract like the gear joint or motors.

This submodule contain all the constraints that are supported by Pymunk.

All the constraints support copy and pickle from the standard library. Custom 
properties set on a constraint will also be copied/pickled.

Chipmunk has a good overview of the different constraint on youtube which
works fine to showcase them in Pymunk as well.
http://www.youtube.com/watch?v=ZgJJZTS0aMM

.. raw:: html

    <iframe width="420" height="315" style="display: block; margin: 0 auto;"
    src="http://www.youtube.com/embed/ZgJJZTS0aMM" frameborder="0"
    allowfullscreen></iframe>


Example::

>>> s = mk.Space()
>>> a, b = mk.Body(10,10), mk.Body(10,10)
>>> c = mk.PivotJoint(a, b, (0,0))
>>> s.add(c)

"""
__docformat__ = "reStructuredText"

__all__ = [
    "Constraint",
    "PinJoint",
    "SlideJoint",
    "PivotJoint",
    "GrooveJoint",
    "DampedSpring",
    "DampedRotarySpring",
    "RotaryLimitJoint",
    "RatchetJoint",
    "GearJoint",
    "SimpleMotor",
]

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from ._chipmunk_cffi import ffi, lib
from ._mixins import PickleMixin
from .util import void, inner_constraints, cffi_body, init_attributes, cp_property
from .vec2d import Vec2d, VecLike, vec2d_from_cffi

if TYPE_CHECKING:
    from .body import Body
    from .space import Space
    import easymunk as mk

SolveFunc = Callable[["Constraint", "Space"], None]

REST_ANGLE = "The relative angle in radians that the bodies want to have"
REST_LENGTH = "The distance the spring wants to be."
STIFFNESS = "The spring constant (Young's modulus)."
DAMPING = "How soft to make the damping of the spring."
T = TypeVar('T')


def constraint_property(attr, doc=None, wrap=None):
    """Wraps cpConstraint[Get|Set]{attr} APIs"""
    return cp_property("Constraint", attr, doc, wrap=wrap)


def anchor_property(name) -> Any:
    """Wraps cp{name}[Get|Set]Anchor[A|B] APIs"""
    return cp_property(
        name[:-1],
        f"Anchor{name[-1]}",
        doc=f"Anchor point in local coordinates of body {name[-1]}.",
        wrap=vec2d_from_cffi,
    )


class Constraint(PickleMixin):
    """Base class of all constraints.

    You usually don't want to create instances of this class directly, but
    instead use one of the specific constraints such as the PinJoint.
    """

    _pickle_args = "a", "b"
    _pickle_kwargs = "max_force", "error_bias", "max_bias", "collide_bodies"
    _pickle_meta_hide = {'_a', '_b', '_cffi_ref', "_cffi_backend", "_nursery",
                         "_cp_post_solve_func", "_cp_pre_solve_func", "_post_solve_func",
                         "_pre_solve_func"}
    _init_kwargs = set(_pickle_args)
    _pre_solve_func: Optional[Callable[["Constraint", "Space"], None]] = None
    _post_solve_func: Optional[Callable[["Constraint", "Space"], None]] = None
    _cp_pre_solve_func: Any = ffi.NULL
    _cp_post_solve_func: Any = ffi.NULL

    max_force: float
    max_force = constraint_property(  # type: ignore
        "MaxForce",
        doc="""The maximum force that the constraint can use to act on the two
        bodies.

        Defaults to infinity
        """,
    )
    error_bias: float
    error_bias = constraint_property(  # type: ignore
        "ErrorBias",
        doc="""The percentage of joint error that remains unfixed after a
        second.

        This works exactly the same as the collision bias property of a space,
        but applies to fixing error (stretching) of joints instead of
        overlapping collisions.

        Defaults to pow(1.0 - 0.1, 60.0) meaning that it will correct 10% of
        the error every 1/60th of a second.
        """,
    )
    max_bias: float
    max_bias = constraint_property(  # type: ignore
        "MaxBias",
        doc="""The maximum speed at which the constraint can apply error
        correction.

        Defaults to infinity
        """,
    )
    collide_bodies: bool
    collide_bodies = constraint_property(  # type: ignore
        "CollideBodies",
        doc="""Constraints can be used for filtering collisions too.

        When two bodies collide, Pymunk ignores the collisions if this property
        is set to False on any constraint that connects the two bodies.
        Defaults to True. This can be used to create a chain that self
        collides, but adjacent links in the chain do not collide.
        """,
    )

    @property
    def impulse(self) -> float:
        """The most recent impulse that constraint applied.

        To convert this to a force, divide by the timestep passed to
        space.step(). You can use this to implement breakable joints to check
        if the force they attempted to apply exceeded a certain threshold.
        """
        return lib.cpConstraintGetImpulse(self._cffi_ref)

    @property
    def a(self) -> "Body":
        """The first of the two bodies constrained"""
        return self._a

    @property
    def b(self) -> "Body":
        """The second of the two bodies constrained"""
        return self._b

    @property
    def pre_solve(self) -> Optional[SolveFunc]:
        """The pre-solve function is called before the constraint solver runs.

        Note that None can be used to reset it to default value.
        """
        return self._pre_solve_func

    @pre_solve.setter
    def pre_solve(self, func: Optional[SolveFunc]):
        self._pre_solve_func = func

        if func is not None:

            @ffi.callback("cpConstraintPreSolveFunc")
            def _impl(_constraint, _space) -> None:
                if self.a.space is None:
                    raise ValueError("body a is not attached to any space")
                fn(self, self.a.space)

            fn = func
            self._cp_pre_solve_func = _impl
        else:
            self._cp_pre_solve_func = ffi.NULL

        lib.cpConstraintSetPreSolveFunc(self._cffi_ref, self._cp_pre_solve_func)

    @property
    def post_solve(self) -> Optional[SolveFunc]:
        """The post-solve function is called after the constraint solver runs.

        Note that None can be used to reset it to default value.
        """
        return self._post_solve_func

    @post_solve.setter
    def post_solve(self, func: Optional[SolveFunc]) -> None:
        self._post_solve_func = func
        if func is not None:

            @ffi.callback("cpConstraintPostSolveFunc")
            def _impl(_constraint, _space) -> None:
                if self.a.space is None:
                    raise ValueError("body a is not attached to any space")
                fn(self, self.a.space)

            fn = func
            self._cp_post_solve_func = _impl
        else:
            self._cp_post_solve_func = ffi.NULL

        lib.cpConstraintSetPostSolveFunc(self._cffi_ref, self._cp_post_solve_func)

    def __init__(self, a: "Body", b: "Body", _constraint: Any, **kwargs) -> None:
        if a is b:
            raise ValueError("cannot apply constraint to same body")

        self._cffi_ref = ffi.gc(_constraint, cffi_free_constraint)
        self._a = a
        self._b = b
        self._nursery = [a, b]
        inner_constraints(a).add(self)
        inner_constraints(b).add(self)
        init_attributes(self, self._init_kwargs, kwargs)

    def __getstate__(self):
        args, meta = super().__getstate__()
        if hasattr(self, '_pre_solve_func'):
            meta['pre_solve'] = self.pre_solve
        if hasattr(self, '_post_solve_func'):
            meta['post_solve'] = self.post_solve
        return args, meta

    def activate_bodies(self) -> None:
        """Activate the bodies this constraint is attached to"""
        self._a.activate()
        self._b.activate()

    def copy(self: T) -> T:
        """Create a deep copy of this object."""
        state = self.__getstate__()
        new = object.__new__(type(self))
        new.__setstate__(state)
        return new


class PinJoint(Constraint):
    """PinJoint links shapes with a solid bar or pin.

    Keeps the anchor points at a set distance from one another.
    """

    _pickle_args = [*Constraint._pickle_args, "anchor_a", "anchor_b"]
    anchor_a: Vec2d = anchor_property("PinJointA")
    anchor_b: Vec2d = anchor_property("PinJointB")
    distance: float
    distance = property(  # type: ignore
        lambda self: lib.cpPinJointGetDist(self._cffi_ref),
        lambda self, distance: void(lib.cpPinJointSetDist(self._cffi_ref, distance)),
        doc="""Fixed distance between anchor points.""",
    )

    def __init__(
            self,
            a: "Body",
            b: "Body",
            anchor_a: VecLike = (0, 0),
            anchor_b: VecLike = (0, 0),
            **kwargs,
    ):
        """a and b are the two bodies to connect, and anchor_a and anchor_b are
        the anchor points on those bodies.

        The distance between the two anchor points is measured when the joint
        is created. If you want to set a specific distance, use the setter
        function to override it.
        """
        ptr = lib.cpPinJointNew(cffi_body(a), cffi_body(b), anchor_a, anchor_b)
        super().__init__(a, b, ptr, **kwargs)


class SlideJoint(Constraint):
    """SlideJoint is like a PinJoint, but have a minimum and maximum distance.

    A chain could be modeled using this joint. It keeps the anchor points
    from getting to far apart, but will allow them to get closer together.
    """

    _pickle_args = [
        *Constraint._pickle_args,
        "anchor_a",
        "anchor_b",
        "min",
        "max",
    ]
    anchor_a: Vec2d = anchor_property("SlideJointA")
    anchor_b: Vec2d = anchor_property("SlideJointB")
    min: float
    min = cp_property(  # type: ignore
        "SlideJoint", "Min", doc="Minimum distance between anchor points."
    )
    max: float
    max = cp_property(  # type: ignore
        "SlideJoint", "Max", doc="Maximum distance between anchor points."
    )

    # noinspection PyShadowingBuiltins
    def __init__(
            self,
            a: "Body",
            b: "Body",
            anchor_a: VecLike,
            anchor_b: VecLike,
            min: float,
            max: float,
            **kwargs,
    ):
        """a and b are the two bodies to connect, anchor_a and anchor_b are the
        anchor points on those bodies, and min and max define the allowed
        distances of the anchor points.
        """
        ref_a = cffi_body(a)
        ref_b = cffi_body(b)
        ptr = lib.cpSlideJointNew(ref_a, ref_b, anchor_a, anchor_b, min, max)
        super().__init__(a, b, ptr, **kwargs)


class PivotJoint(Constraint):
    """PivotJoint allow two objects to pivot about a single point.

    Its like a swivel.
    """

    _pickle_args = [*Constraint._pickle_args, "anchor_a", "anchor_b"]
    anchor_a: Vec2d = anchor_property("PivotJointA")
    anchor_b: Vec2d = anchor_property("PivotJointB")

    def __init__(
            self,
            a: "Body",
            b: "Body",
            *args: VecLike,
            **kwargs,
    ):
        """a and b are the two bodies to connect, and pivot is the point in
        world coordinates of the pivot.

        Because the pivot location is given in world coordinates, you must
        have the bodies moved into the correct positions already.
        Alternatively you can specify the joint based on a pair of anchor
        points, but make sure you have the bodies in the right place as the
        joint will fix itself as soon as you start simulating the space.

        That is, either create the joint with PivotJoint(a, b, pivot) or
        PivotJoint(a, b, anchor_a, anchor_b).

        Args:
            a: The first of the two bodies
            b: The second of the two bodies
            args: Either one pivot point, or two anchor points
        """
        ref_a = cffi_body(a)
        ref_b = cffi_body(b)
        if len(args) == 1:
            ptr = lib.cpPivotJointNew(ref_a, ref_b, args[0])
        elif len(args) == 2:
            ptr = lib.cpPivotJointNew2(ref_a, ref_b, args[0], args[1])
        else:
            msg = "You must specify either one pivot point" " or two anchor points"
            raise TypeError(msg)
        super().__init__(a, b, ptr, **kwargs)


class GrooveJoint(Constraint):
    """GrooveJoint is similar to a PivotJoint, but with a linear slide.

    One of the anchor points is a line segment that the pivot can slide in instead of
    being fixed.
    """

    _pickle_args = [
        *Constraint._pickle_args,
        "groove_a",
        "groove_b",
        "anchor_b",
    ]
    anchor_b: Vec2d = anchor_property("GrooveJointB")
    groove_a: Vec2d
    groove_a = cp_property(  # type: ignore
        "GrooveJoint",
        "GrooveA",
        doc="Start of groove relative to body A.",
        wrap=vec2d_from_cffi,
    )
    groove_b: Vec2d
    groove_b = cp_property(  # type: ignore
        "GrooveJoint",
        "GrooveB",
        doc="Start of groove relative to body B.",
        wrap=vec2d_from_cffi,
    )

    def __init__(
            self,
            a: "Body",
            b: "Body",
            groove_a: VecLike,
            groove_b: VecLike,
            anchor_b: VecLike,
            **kwargs,
    ):
        """The groove goes from groove_a to groove_b on body a, and the pivot
        is attached to anchor_b on body b.

        All coordinates are body local.
        """
        _constraint = lib.cpGrooveJointNew(
            cffi_body(a), cffi_body(b), groove_a, groove_b, anchor_b
        )
        super().__init__(a, b, _constraint, **kwargs)


class DampedSpring(Constraint):
    """DampedSpring is a damped spring.

    The spring allows you to define the rest length, stiffness and damping.
    """

    _pickle_args = [
        *Constraint._pickle_args,
        "anchor_a",
        "anchor_b",
        "rest_length",
        "stiffness",
        "damping",
    ]
    anchor_a: Vec2d = anchor_property("DampedSpringA")
    anchor_b: Vec2d = anchor_property("DampedSpringB")
    rest_length: float = cp_property("DampedSpring", "RestLength", doc=REST_LENGTH)
    stiffness: float = cp_property("DampedSpring", "Stiffness", doc=STIFFNESS)
    damping: float = cp_property("DampedSpring", "Damping", doc=DAMPING)

    def __init__(
            self,
            a: "Body",
            b: "Body",
            anchor_a: VecLike,
            anchor_b: VecLike,
            rest_length: float,
            stiffness: float,
            damping: float,
            **kwargs,
    ):
        """Defined much like a slide joint.

        Args:
            a: Body a
            b: Body b
            anchor_a: Anchor point a, relative to body a
            anchor_b: Anchor point b, relative to body b
            rest_length: The distance the spring wants to be.
            stiffness: The spring constant (Young's modulus).
            damping: How soft to make the damping of the spring.
        """
        ptr = lib.cpDampedSpringNew(
            cffi_body(a),
            cffi_body(b),
            anchor_a,
            anchor_b,
            rest_length,
            stiffness,
            damping,
        )
        super().__init__(a, b, ptr, **kwargs)


class DampedRotarySpring(Constraint):
    """DampedRotarySpring works like the DammpedSpring but in a angular fashion."""

    _pickle_args = [
        *Constraint._pickle_args,
        "rest_angle",
        "stiffness",
        "damping",
    ]
    rest_angle: float = cp_property("DampedRotarySpring", "RestAngle", doc=REST_ANGLE)
    stiffness: float = cp_property("DampedRotarySpring", "Stiffness", doc=STIFFNESS)
    damping: float = cp_property("DampedRotarySpring", "Damping", doc=DAMPING)

    def __init__(
            self,
            a: "Body",
            b: "Body",
            rest_angle: float,
            stiffness: float,
            damping: float,
            **kwargs,
    ):
        """Like a damped spring, but works in an angular fashion.

        Args:
            a: Body a
            b: Body b
            rest_angle: The relative angle in radians that the bodies want to have
            stiffness: The spring constant (Young's modulus).
            damping: How soft to make the damping of the spring.
        """
        ref_a = cffi_body(a)
        ref_b = cffi_body(b)
        ptr = lib.cpDampedRotarySpringNew(ref_a, ref_b, rest_angle, stiffness, damping)
        super().__init__(a, b, ptr, **kwargs)


class RotaryLimitJoint(Constraint):
    """RotaryLimitJoint constrains the relative rotations of two bodies."""

    _pickle_args = [*Constraint._pickle_args, "min", "max"]
    min: float = cp_property("RotaryLimitJoint", "Min")
    max: float = cp_property("RotaryLimitJoint", "Max")

    # noinspection PyShadowingBuiltins
    def __init__(self, a: "Body", b: "Body", min: float, max: float, **kwargs):
        """Constrains the relative rotations of two bodies.

        min and max are the angular limits in radians. It is implemented so
        that it's possible to for the range to be greater than a full
        revolution.
        """
        ptr = lib.cpRotaryLimitJointNew(cffi_body(a), cffi_body(b), min, max)
        super().__init__(a, b, ptr, **kwargs)


class RatchetJoint(Constraint):
    """RatchetJoint is a rotary ratchet, it works like a socket wrench."""

    _pickle_args = [*Constraint._pickle_args, "phase", "ratchet"]

    def __init__(self, a: "Body", b: "Body", phase: float, ratchet: float, **kwargs):
        """Works like a socket wrench.

        ratchet is the distance between "clicks", phase is the initial offset
        to use when deciding where the ratchet angles are.
        """
        ptr = lib.cpRatchetJointNew(cffi_body(a), cffi_body(b), phase, ratchet)
        super().__init__(a, b, ptr, **kwargs)

    angle: float = cp_property("RatchetJoint", "Angle")
    phase: float = cp_property("RatchetJoint", "Phase")
    ratchet: float = cp_property("RatchetJoint", "Ratchet")


class GearJoint(Constraint):
    """GearJoint keeps the angular velocity ratio of a pair of bodies constant."""

    _pickle_args = [*Constraint._pickle_args, "phase", "ratio"]
    phase: float = cp_property("GearJoint", "Phase")
    ratio: float = cp_property("GearJoint", "Ratio")

    def __init__(self, a: "Body", b: "Body", phase: float, ratio: float, **kwargs):
        """Keeps the angular velocity ratio of a pair of bodies constant.

        ratio is always measured in absolute terms. It is currently not
        possible to set the ratio in relation to a third body's angular
        velocity. phase is the initial angular offset of the two bodies.
        """
        ptr = lib.cpGearJointNew(cffi_body(a), cffi_body(b), phase, ratio)
        super().__init__(a, b, ptr, **kwargs)


class SimpleMotor(Constraint):
    """SimpleMotor keeps the relative angular velocity constant."""

    _pickle_args = [*Constraint._pickle_args, "rate"]
    rate: float
    rate = cp_property(  # type: ignore
        "SimpleMotor", "Rate", "Desired relative angular velocity"
    )

    def __init__(self, a: "Body", b: "Body", rate: float, **kwargs):
        """Keeps the relative angular velocity of a pair of bodies constant.

        rate is the desired relative angular velocity. You will usually want
        to set an force (torque) maximum for motors as otherwise they will be
        able to apply a nearly infinite torque to keep the bodies moving.
        """
        ptr = lib.cpSimpleMotorNew(cffi_body(a), cffi_body(b), rate)
        super().__init__(a, b, ptr, **kwargs)


def cffi_free_constraint(cp_constraint) -> None:
    cp_space = lib.cpConstraintGetSpace(cp_constraint)
    if cp_space != ffi.NULL:
        lib.cpSpaceRemoveConstraint(cp_space, cp_constraint)

    logging.debug("free %s", cp_constraint)
    lib.cpConstraintFree(cp_constraint)
