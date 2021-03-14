# ----------------------------------------------------------------------------
# pymunk
# Copyright (c) 2007-2017 Victor Blomqvist
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

"""
Pymunk is a easy-to-use pythonic 2d physics library that can be used whenever
you need 2d rigid body physics from Python.

Homepage: http://www.easymunk.org

This is the main containing module of easymunk. It contains among other things
the very central Space, Body and Shape classes.

"""

__docformat__ = "reStructuredText"

__all__ = [
    "version",
    "chipmunk_version",
    "Space",
    "Body",
    "Shape",
    "Circle",
    "Poly",
    "Segment",
    "moment_for_circle",
    "moment_for_poly",
    "moment_for_segment",
    "moment_for_box",
    "SegmentQueryInfo",
    "ContactPoint",
    "ContactPointSet",
    "Arbiter",
    "CollisionHandler",
    "BB",
    "ShapeFilter",
    "Transform",
    "PointQueryInfo",
    "ShapeQueryInfo",
    "SpaceDebugDrawOptions",
    "Vec2d",
]

from . import _version
from .arbiter import Arbiter
from .bb import BB
from .body import Body
from .collision_handler import CollisionHandler
from .constraints import *
from .contact_point_set import ContactPoint, ContactPointSet
from .geometry import (
    moment_for_circle,
    moment_for_poly,
    moment_for_box,
    moment_for_segment,
    area_for_segment,
    area_for_circle,
    area_for_poly,
)
from .query_info import PointQueryInfo, SegmentQueryInfo, ShapeQueryInfo
from .shape_filter import ShapeFilter
from .shapes import Circle, Poly, Segment, Shape
from .space import Space
from .space_debug_draw_options import SpaceDebugDrawOptions
from .transform import Transform
from .vec2d import Vec2d

version: str = _version.version
"""The release version of this pymunk installation.
Valid only if pymunk was installed from a source or binary
distribution (i.e. not in a checked-out copy from git).
"""

chipmunk_version: str = _version.chipmunk_version
"""The Chipmunk version used with this Pymunk version.

This property does not show a valid value in the compiled documentation, only
when you actually import pymunk and do easymunk.chipmunk_version

The string is in the following format:
<cpVersionString>R<github commit of chipmunk>
where cpVersionString is a version string set by Chipmunk and the git commit
hash corresponds to the git hash of the chipmunk source from
github.com/viblo/Chipmunk2D included with easymunk.
"""

# del cp, ct, u
