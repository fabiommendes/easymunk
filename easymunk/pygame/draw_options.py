# ----------------------------------------------------------------------------
# pymunk
# Copyright (c) 2007-2016 Victor Blomqvist
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

from typing import Sequence, Tuple

import pygame

from ..space_debug_draw_options import SpaceDebugDrawOptions, SpaceDebugColor
from ..vec2d import Vec2d


class DrawOptions(SpaceDebugDrawOptions):
    surface: pygame.Surface

    def __init__(self, surface: pygame.Surface = None, flip_y=False) -> None:
        """Draw a easymunk.Space on a pygame.Surface object.

        Typical usage::

        >>> import easymunk
        >>> surface = pygame.Surface((10,10))
        >>> space = easymunk.Space()
        >>> options = easymunk.pygame.DrawOptions(surface)
        >>> space.debug_draw(options)

        You can control the color of a shape by setting shape.color to the color
        you want it drawn in::

        >>> c = easymunk.Circle(None, 10)
        >>> c.color = pygame.Color("pink")

        See pygame_util.demo.py for a full example

        Since pygame uses a coordiante system where y points down (in contrast
        to many other cases), you either have to make the physics simulation
        with Pymunk also behave in that way, or flip everything when you draw.

        The easiest is probably to just make the simulation behave the same
        way as Pygame does. In that way all coordinates used are in the same
        orientation and easy to reason about::

        >>> space = mk.Space()
        >>> space.gravity = (0, -1000)
        >>> body = space.Body()
        >>> body.position = (0, 0) # will be positioned in the top left corner
        >>> space.debug_draw(options)

        To flip the drawing its possible to set the module property
        :py:data:`positive_y_is_up` to True. Then the pygame drawing will flip
        the simulation upside down before drawing::

        >>> body = mk.Body()
        >>> body.position = (0, 0)
        >>> # Body will be position in bottom left corner

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        if surface is None and pygame.display.get_init():
            surface = pygame.display.get_surface()
        elif surface is None:
            surface = pygame.display.set_mode((800, 600))
        self.surface = surface
        self.flip_y = flip_y
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = self.to_pygame(pos)
        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        p2 = self.to_pygame(circle_edge)
        line_r = 2 if radius > 20 else 1
        pygame.draw.lines(self.surface, outline_color.as_int(), False, [p, p2], line_r)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = self.to_pygame(a)
        p2 = self.to_pygame(b)
        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = self.to_pygame(a)
        p2 = self.to_pygame(b)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [self.to_pygame(v) for v in verts]
        ps += [ps[0]]

        pygame.draw.polygon(self.surface, fill_color.as_int(), ps)

        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, outline_color, outline_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = self.to_pygame(pos)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)

    def mouse_pos(self) -> Vec2d:
        """Get position of the mouse pointer in pymunk coordinates."""
        p = pygame.mouse.get_pos()
        return self.from_pygame(p)

    def to_pygame(self, p: Tuple[float, float], surface=None) -> Vec2d:
        """Convenience method to convert pymunk coordinates to pygame surface
        local coordinates.

        Note that in case positive_y_is_up is False, this function wont actually do
        anything except converting the point to integers.
        """
        if self.flip_y:
            surface = surface or self.surface
            return Vec2d(round(p[0]), surface.get_height() - round(p[1]))
        else:
            return Vec2d(round(p[0]), round(p[1]))

    def from_pygame(self, p: Tuple[float, float]) -> Vec2d:
        """Convenience method to convert pygame surface local coordinates to
        pymunk coordinates
        """
        return self.to_pygame(p)
