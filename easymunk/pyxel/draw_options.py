"""
Draw Pymunk elements using pyxel.
"""
import enum as _enum
from functools import singledispatch, lru_cache, partial

import pyxel

from ..body import Body
from ..shapes import Circle, Segment, Poly
from ..space import Space
from ..space_debug_draw_options import SpaceDebugColor
from ..space_debug_draw_options import SpaceDebugDrawOptions
from ..vec2d import Vec2d

BACKGROUND_COLOR = pyxel.COLOR_BLACK
FOREGROUND_COLOR = pyxel.COLOR_WHITE


class Color(_enum.IntEnum):
    """
    Enum with Pyxel colors
    """

    BLACK = pyxel.COLOR_BLACK
    NAVY = pyxel.COLOR_NAVY
    PURPLE = pyxel.COLOR_PURPLE
    GREEN = pyxel.COLOR_GREEN
    BROWN = pyxel.COLOR_BROWN
    DARKBLUE = pyxel.COLOR_DARKBLUE
    LIGHTBLUE = pyxel.COLOR_LIGHTBLUE
    WHITE = pyxel.COLOR_WHITE
    RED = pyxel.COLOR_RED
    ORANGE = pyxel.COLOR_ORANGE
    YELLOW = pyxel.COLOR_YELLOW
    LIME = pyxel.COLOR_LIME
    CYAN = pyxel.COLOR_CYAN
    GRAY = pyxel.COLOR_GRAY
    PINK = pyxel.COLOR_PINK
    PEACH = pyxel.COLOR_PEACH


ALL_COLORS = [*Color]


class DrawOptions(SpaceDebugDrawOptions):
    def __init__(self, mod=pyxel, keep_shape_colors=True, wireframe=False):
        self.palette = [self.to_hex_color(c) for c in pyxel.DEFAULT_PALETTE]
        self._line = mod.line
        self._circb = mod.circb
        self._circ = mod.circ
        self._pset = mod.pset
        self._draw_path = partial(draw_closed_path, mod=mod)
        self._draw_poly = partial(draw_filled_poly, mod=mod)
        self.keep_shape_colors = keep_shape_colors
        self.wireframe = wireframe
        color_map = {c: i for i, c in enumerate(self.palette)}

        @lru_cache(None)
        def to_color(color):
            if isinstance(color, int):
                return color
            try:
                return int(color_map[color])
            except KeyError:
                return int(color.r % 16)

        self._to_color = to_color
        self.shape_dynamic_color = self.palette[Color.WHITE]
        self.shape_static_color = self.palette[Color.DARKBLUE]
        self.shape_kinematic_color = self.palette[Color.CYAN]
        self.shape_sleeping_color = self.palette[Color.PEACH]
        super().__init__()

        self.shape_outline_color = self.palette[Color.GRAY]
        self.constraint_color = self.palette[Color.RED]
        self.collision_point_color = self.palette[Color.YELLOW]
        self.printed = False

    @staticmethod
    def to_hex_color(u):
        return SpaceDebugColor(u // (256 * 256), u // 256 % 256, u % 256, 255)

    def draw_circle(self, pos, angle, r, col1, col2):
        if self.wireframe:
            endpos = pos + Vec2d(r, 0).rotated(angle)
            col = self._to_color(col2)
            self._circb(*pos, r, col)
            self._line(*pos, *endpos, col)
        else:
            self._circ(*pos, r, self._to_color(col1))

    def draw_segment(self, a, b, col):
        col = self._to_color(col)
        self._line(*a, *b, col)

    def draw_fat_segment(self, a, b, r, col1, col2):
        col = self._to_color(col2)
        self._line(*a, *b, col)

    def draw_polygon(self, pts, r, col1, col2):
        if self.wireframe:
            self._draw_path(pts, self._to_color(col2))
        else:
            self._draw_poly(pts, self._to_color(col1))

    def draw_dot(self, size: int, pt, col):
        self._circ(*pt, size // 2, self._to_color(col))


#
# Background/foreground drawing
#
def bg(col=None) -> int:
    """
    Get or set the default background color for space objects.
    """
    global BACKGROUND_COLOR

    if col is None:
        return BACKGROUND_COLOR
    else:
        BACKGROUND_COLOR = int(col)
        return BACKGROUND_COLOR


def fg(col=None) -> int:
    """
    Get or set the default foreground color for space objects.
    """
    global FOREGROUND_COLOR

    if col is None:
        return FOREGROUND_COLOR
    else:
        FOREGROUND_COLOR = int(col)
        return FOREGROUND_COLOR


#
# Draw Pymunk shapes
#
@singledispatch
def draw(shape, col=None, mod=pyxel):
    """
    Draw Pymunk shape or all shapes in a Pymunk body or space with a given
    offset.

    Args:
        shape: A Pymunk shape, body or space
        mod: x coordinate offset
        col (int): A color index
    """
    try:
        method = shape.draw
    except AttributeError:
        name = type(shape).__name__
        raise TypeError(f"Cannot draw {name} objects")
    else:
        return method(mod, col=col)


@singledispatch
def drawb(shape, col=None, mod=pyxel):
    """
    Like draw, but renders only the outline of a shape.

    Args:
        shape: A Pymunk shape, body or space
        mod: mod namespace  that transforms the scene.
        col (int): A color index
    """
    try:
        method = shape.drawb
    except AttributeError:
        name = type(shape).__name__
        raise TypeError(f"Cannot draw {name} objects")
    else:
        return method(mod, col=col)


#
# Register implementation for draw() and drawb() functions
#
@draw.register(Space)
def draw_space(s: Space, col=None, mod=pyxel):
    if hasattr(s, "background_color"):
        pyxel.cls(s.background_color)
    elif BACKGROUND_COLOR is not None:
        pyxel.cls(BACKGROUND_COLOR)
    for a in s.shapes:
        draw(a, col, mod)


@draw.register(Body)
def draw_body(b: Body, col=None, mod=pyxel):
    for s in b.shapes:
        draw(s, col, mod)


@draw.register(Circle)
def draw_circle(s: Circle, col=None, mod=pyxel):
    color = getattr(s, "color", None)
    if color is None:
        color = FOREGROUND_COLOR if col is None else col
    mod.circ(*(s.body.position + s.offset), s.radius, color)


@draw.register(Segment)
def draw_segment(s: Segment, col=None, mod=pyxel):
    (x1, y1), (x2, y2) = map(s.body.local_to_world, [s.a, s.b])
    color = getattr(s, "color", None)
    if color is None:
        color = FOREGROUND_COLOR if col is None else col
    mod.line(x1, y1, x2, y2, color)


@draw.register(Poly)
def draw_poly(s: Poly, col=None, mod=pyxel):
    vertices = [s.body.local_to_world(v) for v in s.get_vertices()]
    color = getattr(s, "color", None)
    if color is None:
        color = FOREGROUND_COLOR if col is None else col
    return draw_filled_poly(vertices, mod, color)


def draw_filled_poly(vertices, col=None, mod=pyxel):
    n = len(vertices)
    if n == 1:
        x, y = vertices[0]
        mod.pset(x, y, col)
        return
    elif n == 2:
        (x1, y1), (x2, y2) = vertices
        mod.line(x1, x2, y1, y2, col)
        return

    (x1, y1), (x2, y2), (x3, y3), *rest = vertices
    mod.tri(x1, y1, x2, y2, x3, y3, col)
    rest.reverse()

    while rest:
        x2, y2 = x3, y3
        x3, y3 = rest.pop()
        mod.tri(x1, y1, x2, y2, x3, y3, col)


def draw_path(path, col=None, mod=pyxel):
    a, *rest = path
    for b in path:
        mod.line(*a, *b, col)
        a = b


def draw_closed_path(path, col, mod=pyxel):
    draw_path(path, col, mod)
    mod.line(*path[0], *path[-1], col)


@drawb.register(Space)
def drawb_space(s: Space, col=None, mod=pyxel):
    for a in s.shapes:
        drawb(a, col, mod)


@drawb.register(Body)
def drawb_body(b: Body, col=None, mod=pyxel):
    for s in b.shapes:
        drawb(s, col, mod)


@drawb.register(Circle)
def drawb_circle(s: Circle, col=None, mod=pyxel):
    color = getattr(s, "color", None)
    if color is None:
        color = FOREGROUND_COLOR if col is None else col
    mod.circb(*(s.body.position + s.offset), s.radius, color)


@drawb.register(Poly)
def drawb_poly(s: Poly, col=None, mod=pyxel):
    vertices = [s.body.local_to_world(v) for v in s.get_vertices()]
    vertices.append(vertices[0])
    color = getattr(s, "color", None)
    if color is None:
        color = FOREGROUND_COLOR if col is None else col
    return draw_path(vertices, mod, color)


drawb.register(Segment, draw_segment)
