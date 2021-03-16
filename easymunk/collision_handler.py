__version__ = "$Id$"
__docformat__ = "reStructuredText"

import functools
import warnings
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import sidekick.api as sk

from ._chipmunk_cffi import ffi
from .arbiter import Arbiter
from .util import void

if TYPE_CHECKING:
    from .space import Space

BoolCB = Callable[[Arbiter, "Space", Any], bool]
NullCB = Callable[[Arbiter, "Space", Any], None]
Ptr = ffi.CData
CFFI_REF_ATTR = {
    "begin": "beginFunc",
    "pre_solve": "preSolveFunc",
    "post_solve": "postSolveFunc",
    "separate": "separateFunc",
}
CFFI_FUNC_TYPE = {
    "begin": "cpCollisionBeginFunc",
    "pre_solve": "cpCollisionPreSolveFunc",
    "post_solve": "cpCollisionPostSolveFunc",
    "separate": "cpCollisionSeparateFunc",
}


def always_collide(arb, space, data):
    return True


def do_nothing(arb, space, data):
    return None


class CollisionHandler:
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

    space = sk.alias("_space")
    begin: Optional[BoolCB]
    begin = property(  # type: ignore
        lambda self: self._begin_base,
        lambda self, cb: void(self._set_cb("begin", self._bool_cb, cb)),
        doc="""Two shapes just started touching for the first time this step.

        ``func(arbiter, space, data) -> bool``

        Return true from the callback to process the collision normally or
        false to cause pymunk to ignore the collision entirely. If you return
        false, the `pre_solve` and `post_solve` callbacks will never be run,
        but you will still recieve a separate event when the shapes stop
        overlapping.
        """,
    )
    pre_solve: Optional[BoolCB]
    pre_solve = property(  # type: ignore
        lambda self: self._pre_solve_base,
        lambda self, cb: void(self._set_cb("pre_solve", self._bool_cb, cb)),
        doc="""Two shapes are touching during this step.

        ``func(arbiter, space, data) -> bool``

        Return false from the callback to make pymunk ignore the collision
        this step or true to process it normally. Additionally, you may
        override collision values using Arbiter.friction, Arbiter.elasticity
        or Arbiter.surfaceVelocity to provide custom friction, elasticity,
        or surface velocity values. See Arbiter for more info.
        """,
    )
    post_solve: Optional[NullCB]
    post_solve = property(  # type: ignore
        lambda self: self._post_solve_base,
        lambda self, cb: void(self._set_cb("post_solve", self._null_cb, cb)),
        doc="""Two shapes are touching and their collision response has been
        processed.

        ``func(arbiter, space, data)``

        You can retrieve the collision impulse or kinetic energy at this
        time if you want to use it to calculate sound volumes or damage
        amounts. See Arbiter for more info.
        """,
    )
    separate: Optional[NullCB]
    separate = property(  # type: ignore
        lambda self: self._separate_base,
        lambda self, cb: void(self._set_cb("separate", self._locked_cb, cb)),
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
        self._cffi_ref = _handler
        self._space = space
        self._begin = None
        self._begin_base: Optional[BoolCB] = None  # For pickle
        self._pre_solve = None
        self._pre_solve_base: Optional[BoolCB] = None  # For pickle
        self._post_solve = None
        self._post_solve_base: Optional[NullCB] = None  # For pickle
        self._separate = None
        self._separate_base: Optional[NullCB] = None  # For pickle
        self._data: Dict[Any, Any] = {}

    def as_dict(self) -> Dict[str, callable]:
        """
        Return handler state as a dictionary
        """
        return {
            "begin": self.begin,
            "pre_solve": self.pre_solve,
            "post_solve": self.post_solve,
            "separate": self.separate,
        }

    def update(self, data=MappingProxyType({}), **kwargs) -> None:
        """
        Update handler functions from dictionary or keyword arguments.
        """
        for k, v in {**data, **kwargs}.items():
            setattr(self, k, v)

    def _reset(self) -> None:
        self.begin = always_collide
        self.pre_solve = always_collide
        self.post_solve = do_nothing
        self.separate = do_nothing

    def _set_cb(self, name, factory, func) -> None:
        attr = CFFI_REF_ATTR[name]
        cb_type = CFFI_FUNC_TYPE[name]
        cf = functools.partial(factory, func or always_collide)
        ptr = ffi.callback(cb_type)(cf)

        setattr(self, f"_{name}_base", func)
        setattr(self, f"_{name}", ptr)
        setattr(self._cffi_ref, attr, ptr)

    def _bool_cb(self, func: BoolCB, ptr: Ptr, space: Ptr, data: Ptr) -> bool:
        arb = Arbiter(ptr, self._space)
        out = func(arb, self._space, self._data)

        if isinstance(out, bool):
            return out

        code = getattr(func, "__code__", None)
        func_name = getattr(func, "__name__", "<function>")
        filename = getattr(code, "co_filename", "<unknown file>")
        lineno = getattr(code, "co_firstlineno", -1)
        module = getattr(func, "__module__", "<string>")

        msg = (
            f"Function '{func_name}' should return a bool to"
            " indicate if the collision should be processed or not when"
            " used as 'begin' or 'pre_solve' collision callback."
        )
        warnings.warn_explicit(msg, UserWarning, filename, lineno, module)
        return True

    def _null_cb(self, func: NullCB, ptr: Ptr, space: Ptr, data: Ptr) -> None:
        arb = Arbiter(ptr, self._space)
        func(arb, self._space, self._data)

    def _locked_cb(self, func: NullCB, ptr: Ptr, space: Ptr, data: Ptr) -> None:
        # this try is needed since a separate callback will be called
        # if a colliding object is removed, regardless if its in a
        # step or not.
        with self._space.locked():
            arb = Arbiter(ptr, self._space)
            func(arb, self._space, self._data)
