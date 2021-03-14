__docformat__ = "reStructuredText"

import math
from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence, Tuple, Type

from ._chipmunk_cffi import ffi, lib
from .body import Body
from .space import shape_from_cffi
from .vec2d import Vec2d

if TYPE_CHECKING:
    from .shapes import Shape
    from types import TracebackType

_DrawFlags = int


class SpaceDebugColor(NamedTuple):
    """Color tuple used by the debug drawing API."""

    r: float
    g: float
    b: float
    a: float = 255

    def as_int(self) -> Tuple[int, int, int, int]:
        """Return the color as a tuple of ints, where each value is rounded.

        >>> SpaceDebugColor(0, 51.1, 101.9, 255).as_int()
        (0, 51, 102, 255)
        """
        return round(self[0]), round(self[1]), round(self[2]), round(self[3])

    def as_float(self) -> Tuple[float, float, float, float]:
        """Return the color as a tuple of floats, each value divided by 255.

        >>> SpaceDebugColor(0, 51, 102, 255).as_float()
        (0.0, 0.2, 0.4, 1.0)
        """
        return self[0] / 255.0, self[1] / 255.0, self[2] / 255.0, self[3] / 255.0


class SpaceDebugDrawOptions:
    """SpaceDebugDrawOptions configures debug drawing.

    If appropriate its usually easy to use the supplied draw implementations
    directly: easymunk.pygame_util, easymunk.pyglet_util and easymunk.matplotlib_util.
    """

    DRAW_SHAPES = lib.CP_SPACE_DEBUG_DRAW_SHAPES
    """Draw shapes.  
    
    Use on the flags property to control if shapes should be drawn or not.
    """

    DRAW_CONSTRAINTS = lib.CP_SPACE_DEBUG_DRAW_CONSTRAINTS
    """Draw constraints. 
    
    Use on the flags property to control if constraints should be drawn or not.
    """

    DRAW_COLLISION_POINTS = lib.CP_SPACE_DEBUG_DRAW_COLLISION_POINTS
    """Draw collision points.
    
    Use on the flags property to control if collision points should be drawn or
    not.
    """

    shape_dynamic_color = SpaceDebugColor(52, 152, 219, 255)
    shape_static_color = SpaceDebugColor(149, 165, 166, 255)
    shape_kinematic_color = SpaceDebugColor(39, 174, 96, 255)
    shape_sleeping_color = SpaceDebugColor(114, 148, 168, 255)
    shape_outline_color: SpaceDebugColor
    shape_outline_color = property(  # type: ignore
        lambda self: self._to_color(self._cffi_ref.shapeOutlineColor),
        lambda self, c: setattr(self._cffi_ref, "shapeOutlineColor", c),
        doc="""The outline color of shapes.
        
        Should be a tuple of 4 ints between 0 and 255 (r, g, b, a).
        """,
    )
    constraint_color: SpaceDebugColor
    constraint_color = property(  # type: ignore
        lambda self: self._to_color(self._cffi_ref.constraintColor),
        lambda self, c: setattr(self._cffi_ref, "constraintColor", c),
        doc="""The color of constraints.

        Should be a tuple of 4 ints between 0 and 255 (r, g, b, a).
        """,
    )
    collision_point_color: SpaceDebugColor
    collision_point_color = property(  # type: ignore
        lambda self: self._to_color(self._cffi_ref.collisionPointColor),
        lambda self, c: setattr(self._cffi_ref, "collisionPointColor", c),
        doc="""The color of collisions.

        Should be a tuple of 4 ints between 0 and 255 (r, g, b, a).
        """,
    )
    flags: _DrawFlags
    flags = property(  # type: ignore
        lambda self: self._cffi_ref.flags,
        lambda self, f: setattr(self._cffi_ref, "flags", f),
        doc="""Bit flags which of shapes, joints and collisions should be drawn.
    
        By default all 3 flags are set, meaning shapes, joints and collisions 
        will be drawn.

        Example using the basic text only DebugDraw implementation (normally
        you would the desired backend instead, such as 
        `pygame_util.DrawOptions` or `pyglet_util.DrawOptions`):
        """,
    )

    def __init__(self, bypass_chipmunk=False) -> None:
        ptr = ffi.new("cpSpaceDebugDrawOptions *")
        self._cffi_ref = ptr
        self.shape_outline_color = SpaceDebugColor(44, 62, 80, 255)
        self.constraint_color = SpaceDebugColor(142, 68, 173, 255)
        self.collision_point_color = SpaceDebugColor(231, 76, 60, 255)

        # Set to false to bypass chipmunk shape drawing code
        self.bypass_chipmunk = bypass_chipmunk
        self.flags = (
                SpaceDebugDrawOptions.DRAW_SHAPES
                | SpaceDebugDrawOptions.DRAW_CONSTRAINTS
                | SpaceDebugDrawOptions.DRAW_COLLISION_POINTS
        )
        self._callbacks = cffi_register_debug_draw_options_callbacks(self, ptr)

    def __enter__(self) -> None:
        pass

    def __exit__(
            self,
            typ: Optional[Type[BaseException]],
            value: Optional[BaseException],
            traceback: Optional["TracebackType"],
    ) -> None:
        pass

    def _to_color(self, color: ffi.CData) -> SpaceDebugColor:
        return SpaceDebugColor(color.r, color.g, color.b, color.a)

    def _print(self, *args, **kwargs):
        return print(*args, **kwargs)

    def draw_circle(
            self,
            pos: Vec2d,
            angle: float,
            radius: float,
            outline_color: SpaceDebugColor,
            fill_color: SpaceDebugColor,
    ) -> None:
        self._print("draw_circle", (pos, angle, radius, outline_color, fill_color))

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        self._print("draw_segment", (a, b, color))

    def draw_fat_segment(
            self,
            a: Vec2d,
            b: Vec2d,
            radius: float,
            outline_color: SpaceDebugColor,
            fill_color: SpaceDebugColor,
    ) -> None:
        self._print("draw_fat_segment", (a, b, radius, outline_color, fill_color))

    def draw_polygon(
            self,
            verts: Sequence[Vec2d],
            radius: float,
            outline_color: SpaceDebugColor,
            fill_color: SpaceDebugColor,
    ) -> None:
        self._print("draw_polygon", (verts, radius, outline_color, fill_color))

    def draw_dot(self, size: float, pos: Vec2d, color: SpaceDebugColor) -> None:
        self._print("draw_dot", (size, pos, color))

    def draw_shape(self, shape: "Shape") -> None:
        self._print("draw_shape", shape)

    def color_for_shape(self, shape: "Shape") -> SpaceDebugColor:
        if hasattr(shape, "color"):
            return SpaceDebugColor(*shape.color)  # type: ignore

        color = self.shape_dynamic_color
        if shape.body is not None:
            if shape.body.body_type == Body.STATIC:
                color = self.shape_static_color
            elif shape.body.body_type == Body.KINEMATIC:
                color = self.shape_kinematic_color
            elif shape.body.is_sleeping:
                color = self.shape_sleeping_color

        return color


def color_from_cffi(color: ffi.CData) -> SpaceDebugColor:
    return SpaceDebugColor(color.r, color.g, color.b, color.a)


def cffi_register_debug_draw_options_callbacks(opts: SpaceDebugDrawOptions, ptr):
    @ffi.callback("cpSpaceDebugDrawCircleImpl")
    def f1(pos, angle, radius, outline, fill, _):
        pos = Vec2d(pos.x, pos.y)
        c1 = color_from_cffi(outline)
        c2 = color_from_cffi(fill)
        opts.draw_circle(pos, angle, radius, c1, c2)

    @ffi.callback("cpSpaceDebugDrawSegmentImpl")
    def f2(a, b, color, _):  # type: ignore
        # sometimes a and/or b can be nan. For example if both endpoints
        # of a spring is at the same position. In those cases skip calling
        # the drawing method.
        if math.isnan(a.x) or math.isnan(a.y) or math.isnan(b.x) or math.isnan(b.y):
            return
        opts.draw_segment(Vec2d(a.x, a.y), Vec2d(b.x, b.y), color_from_cffi(color))

    @ffi.callback("cpSpaceDebugDrawFatSegmentImpl")
    def f3(a, b, radius, outline, fill, _):
        a = Vec2d(a.x, a.y)
        b = Vec2d(b.x, b.y)
        opts.draw_fat_segment(a, b, radius, *map(color_from_cffi, (outline, fill)))

    @ffi.callback("cpSpaceDebugDrawPolygonImpl")
    def f4(count, verts, radius, outline, fill, _):
        vs = []
        for i in range(count):
            vs.append(Vec2d(verts[i].x, verts[i].y))
        opts.draw_polygon(vs, radius, color_from_cffi(outline), color_from_cffi(fill))

    @ffi.callback("cpSpaceDebugDrawDotImpl")
    def f5(size, pos, color, _):
        opts.draw_dot(size, Vec2d(pos.x, pos.y), color_from_cffi(color))

    @ffi.callback("cpSpaceDebugDrawColorForShapeImpl")
    def f6(shape, data):
        space = ffi.from_handle(data)
        return opts.color_for_shape(shape_from_cffi(space, shape))

    ptr.drawCircle = f1
    ptr.drawSegment = f2
    ptr.drawFatSegment = f3
    ptr.drawPolygon = f4
    ptr.drawDot = f5
    ptr.colorForShape = f6

    return [f1, f2, f3, f4, f5, f6]
