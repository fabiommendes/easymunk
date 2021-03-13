__version__ = "$Id$"
__docformat__ = "reStructuredText"

import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from ._chipmunk_cffi import ffi
from .arbiter import Arbiter
from .util import void

if TYPE_CHECKING:
    from .space import Space

_CollisionCallbackBool = Callable[[Arbiter, "Space", Any], bool]
_CollisionCallbackNoReturn = Callable[[Arbiter, "Space", Any], None]


class CollisionHandler(object):
    """A collision handler is a set of 4 function callbacks for the different
    collision events that Pymunk recognizes.

    Collision callbacks are closely associated with Arbiter objects. You
    should familiarize yourself with those as well.

    Note #1: Shapes tagged as sensors (Shape.sensor == true) never generate
    collisions that get processed, so collisions between sensors shapes and
    other shapes will never call the post_solve() callback. They still
    generate begin(), and separate() callbacks, and the pre_solve() callback
    is also called every frame even though there is no collision response.
    Note #2: pre_solve() callbacks are called before the sleeping algorithm
    runs. If an object falls asleep, its post_solve() callback won't be
    called until it's re-awoken.
    """

    @property
    def data(self) -> Dict[Any, Any]:
        """Data property that get passed on into the
        callbacks.

        data is a dictionary and you can not replace it, only fill it with data.

        Useful if the callback needs some extra data to perform its function.
        """
        return self._data

    begin: Optional[_CollisionCallbackBool]
    begin = property(  # type: ignore
        lambda self: self._begin_base,
        lambda self, cb: void(self.__set_begin_cb(cb)),
        doc="""Two shapes just started touching for the first time this step.

        ``func(arbiter, space, data) -> bool``

        Return true from the callback to process the collision normally or
        false to cause pymunk to ignore the collision entirely. If you return
        false, the `pre_solve` and `post_solve` callbacks will never be run,
        but you will still recieve a separate event when the shapes stop
        overlapping.
        """,
    )
    pre_solve: Optional[_CollisionCallbackBool]
    pre_solve = property(  # type: ignore
        lambda self: self._pre_solve_base,
        lambda self, cb: void(self.__set_pre_solve_cb(cb)),
        doc="""Two shapes are touching during this step.

        ``func(arbiter, space, data) -> bool``

        Return false from the callback to make pymunk ignore the collision
        this step or true to process it normally. Additionally, you may
        override collision values using Arbiter.friction, Arbiter.elasticity
        or Arbiter.surfaceVelocity to provide custom friction, elasticity,
        or surface velocity values. See Arbiter for more info.
        """,
    )
    post_solve: Optional[_CollisionCallbackNoReturn]
    post_solve = property(  # type: ignore
        lambda self: self._post_solve_base,
        lambda self, cb: void(self.__set_post_solve_cb(cb)),
        doc="""Two shapes are touching and their collision response has been
        processed.

        ``func(arbiter, space, data)``

        You can retrieve the collision impulse or kinetic energy at this
        time if you want to use it to calculate sound volumes or damage
        amounts. See Arbiter for more info.
        """,
    )
    separate: Optional[_CollisionCallbackNoReturn]
    separate = property(  # type: ignore
        lambda self: self._separate_base,
        lambda self, cb: void(self.__set_separate_cb(cb)),
        doc="""Two shapes have just stopped touching for the first time this
        step.

        ``func(arbiter, space, data)``

        To ensure that begin()/separate() are always called in balanced
        pairs, it will also be called when removing a shape while its in
        contact with something or when de-allocating the space.
        """,
    )

    def __init__(self, _handler: Any, space: "Space") -> None:
        """Initialize a CollisionHandler object from the Chipmunk equivalent
        struct and the Space.

        .. note::
            You should never need to create an instance of this class directly.
        """
        self._handler = _handler
        self._space = space
        self._begin = None
        self._begin_base: Optional[_CollisionCallbackBool] = None  # For pickle
        self._pre_solve = None
        self._pre_solve_base: Optional[_CollisionCallbackBool] = None  # For pickle
        self._post_solve = None
        self._post_solve_base: Optional[_CollisionCallbackNoReturn] = None  # For pickle
        self._separate = None
        self._separate_base: Optional[_CollisionCallbackNoReturn] = None  # For pickle

        self._data: Dict[Any, Any] = {}

    def _reset(self) -> None:
        always_collide = lambda arb, space, data: True
        do_nothing = lambda arb, space, data: None

        self.begin = always_collide
        self.pre_solve = always_collide
        self.post_solve = do_nothing
        self.separate = do_nothing

    def __set_begin_cb(self, func) -> None:
        @ffi.callback("cpCollisionBeginFunc")
        def cf(_arb: ffi.CData, _space: ffi.CData, _: ffi.CData) -> bool:
            out = func(Arbiter(_arb, self._space), self._space, self._data)
            if isinstance(out, bool):
                return out

            func_name = func.__code__.co_name
            filename = func.__code__.co_filename
            lineno = func.__code__.co_firstlineno

            msg = (
                f"Function '{func_name}' should return a bool to"
                " indicate if the collision should be processed or not when"
                " used as 'begin' or 'pre_solve' collision callback."
            )
            warnings.warn_explicit(msg, UserWarning, filename, lineno, func.__module__)
            return True

        self._begin = cf
        self._begin_base = func
        self._handler.beginFunc = cf

    def __set_pre_solve_cb(self, func) -> None:
        @ffi.callback("cpCollisionPreSolveFunc")
        def cf(_arb: ffi.CData, _space: ffi.CData, _: ffi.CData) -> bool:
            out = func(Arbiter(_arb, self._space), self._space, self._data)
            if isinstance(out, bool):
                return out

            func_name = func.__code__.co_name
            filename = func.__code__.co_filename
            lineno = func.__code__.co_firstlineno

            msg = (
                f"Function '{func_name}' should return a bool to indicate if the "
                f"collision should be processed or not when used as 'begin' or "
                f"'pre_solve' collision callback."
            )
            warnings.warn_explicit(msg, UserWarning, filename, lineno, func.__module__)
            return True

        self._pre_solve = cf
        self._pre_solve_base = func
        self._handler.preSolveFunc = cf

    def __set_post_solve_cb(self, func: _CollisionCallbackNoReturn) -> None:
        @ffi.callback("cpCollisionPostSolveFunc")
        def cf(_arb: ffi.CData, _space: ffi.CData, _: ffi.CData) -> None:
            func(Arbiter(_arb, self._space), self._space, self._data)

        self._post_solve = cf
        self._post_solve_base = func
        self._handler.postSolveFunc = cf

    def __set_separate_cb(self, func):
        @ffi.callback("cpCollisionSeparateFunc")
        def cf(_arb: ffi.CData, _space: ffi.CData, _: ffi.CData) -> None:
            try:
                # this try is needed since a separate callback will be called
                # if a colliding object is removed, regardless if its in a
                # step or not.
                self._space._locked = True
                func(Arbiter(_arb, self._space), self._space, self._data)
            finally:
                self._space._locked = False

        self._separate = cf
        self._separate_base = func
        self._handler.separateFunc = cf
