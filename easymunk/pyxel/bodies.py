"""
Easier factory functions for creating Pymunk objects.
"""
from functools import wraps

import pyxel

from . import Color, DrawOptions
from . import draw_mods
from ..body import CircleBody, SegmentBody, PolyBody, Body
from ..space import Space
from ..vec2d import Vec2d

DEFAULT_SPACE = None
MOMENT_MULTIPLIER = 5.0


def body_maker(func):
    """
    Decorate function that normalize input arguments and outputs for a pyxel
    context.
    """

    @wraps(func)
    def maker(*args, **kwargs):
        kwargs.setdefault("space", DEFAULT_SPACE)
        kwargs.setdefault("elasticity", 1.0)
        col = kwargs.pop("col", None)
        if col:
            kwargs["color"] = col
        body = func(*args, **kwargs)
        if "moment" not in kwargs:
            body.moment *= MOMENT_MULTIPLIER
        return body

    return maker


#
# Basic geometric shapes
#
@body_maker
def circ(x: float, y: float, r: float, **kwargs) -> CircleBody:
    """
    Creates a body with a Circle shape attached to it.

    Args:
        x: Center point x coordinate
        y: Center point y coordinate
        r: Circle radius
    """
    return CircleBody(r, position=(x, y), **kwargs)


@body_maker
def line(
    x1: float, y1: float, x2: float, y2: float, radius: float = 1.0, **kwargs
) -> SegmentBody:
    """
    Creates a body with a Segment shape attached to it.

    Args:
        x1: x coordinate of starting point
        y1: y coordinate of starting point
        x2: x coordinate of ending point
        y2: y coordinate of ending point
        radius (float): Collision radius for line element.
    """
    a, b = Vec2d(x1, y1), Vec2d(x2, y2)
    cm = (a + b) / 2
    return SegmentBody(a - cm, b - cm, radius=radius, **kwargs)


@body_maker
def tri(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    radius: float = 0.0,
    **kwargs
) -> PolyBody:
    """
    Creates a Pymunk body with a triangular Poly shape attached to it.

    Args:
        x1: x coordinate of first point
        y1: y coordinate of first point
        x2: x coordinate of second point
        y2: y coordinate of second point
        x3: x coordinate of last point
        y3: y coordinate of last point
        radius: Collision radius for line element.
    """
    x = (x1 + x2 + x3) / 3
    y = (y1 + y2 + y3) / 3
    vertices = [Vec2d(x1 - x, y1 - y), Vec2d(x2 - x, y2 - y), Vec2d(x3 - x, y3 - y)]
    return PolyBody(vertices, radius=radius, **kwargs)


@body_maker
def rect(
    x: float, y: float, w: float, h: float, radius: float = 0.0, **kwargs
) -> PolyBody:
    """
    Creates a Pymunk body with a triangular Poly shape attached to it.

    Args:
        x: x coordinate of starting point
        y: y coordinate of starting point
        w: width
        h: height
        radius: Collision radius for line element.
    """
    x_ = x + w / 2
    y_ = y + h / 2
    return PolyBody.new_box((w, h), position=(x_, y_), radius=radius, **kwargs)


@body_maker
def margin(
    x: int = 0, y: int = 0, width: int = None, height: int = None, **kwargs
) -> Body:
    """
    Creates a margin around the screen.
    """
    if width is None:
        width = pyxel.width - 1
    if height is None:
        height = pyxel.height - 1
    a, b, c, d = (x, y), (x + width, y), (x + width, y + height), (x, y + height)

    # noinspection PyProtectedMember
    opts = {k: kwargs.pop(k) for k in Body._init_kwargs if k in kwargs}
    body = Body(body_type=Body.STATIC, **opts)
    body.create_segment(a, b, **kwargs)
    body.create_segment(b, c, **kwargs)
    body.create_segment(c, d, **kwargs)
    body.create_segment(d, a, **kwargs)
    return body


def space(
    bg: Color = pyxel.COLOR_BLACK,
    col: Color = pyxel.COLOR_WHITE,
    mod=pyxel,
    flip_y: bool = False,
    wireframe: bool = False,
    **kwargs
):
    """
    Create a space object.

    Args:
        bg: Background color.
        col: Default foreground color.
        mod: Pyxel module or some module-like object with corresponding draw functions.
        flip_y: Flip y coordinate, making coordinates consistent with mathematical
            convention.
        wireframe: Draw shapes in wireframe mode.
    """
    global DEFAULT_SPACE
    DEFAULT_SPACE = sp = Space(**kwargs)

    if flip_y:
        draw_options = DrawOptions(draw_mods.flip_y)
    else:
        draw_options = DrawOptions(mod)
    draw_options.wireframe = wireframe

    def update(dt: float = 1 / getattr(pyxel, "DEFAULT_FPS", 30)):
        sp.step(dt)

    def draw(clear: bool = False):
        if clear:
            pyxel.cls(sp.bg)
        sp.debug_draw(draw_options)

    # noinspection PyShadowingNames
    def run(wireframe=wireframe):
        """
        Run pyxel engine alongside with physics.
        """
        draw_options.wireframe = wireframe
        pyxel.run(update, lambda: draw(clear=True))

    sp.col = col
    sp.bg = bg
    sp.update = update
    sp.draw = draw
    sp.run = run
    return sp


def moment_multiplier(value: float = None) -> float:
    """
    Default multiplier used to calculate the moment of standard shapes.

    Call with argument to set value, and return value if called with no
    arguments.
    """
    global MOMENT_MULTIPLIER

    if value is None:
        return MOMENT_MULTIPLIER
    else:
        MOMENT_MULTIPLIER = value
        return value
