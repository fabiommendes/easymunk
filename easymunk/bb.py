__docformat__ = "reStructuredText"

from typing import NamedTuple, Iterable

from . import _chipmunk_cffi
from .vec2d import Vec2d, VecLike

lib = _chipmunk_cffi.lib
ffi = _chipmunk_cffi.ffi


class BB(NamedTuple):
    """Simple axis-aligned 2D bounding box.

    Stored as left, bottom, right, top values.

    An instance can be created in this way:
        >>> BB(left=1, bottom=5, right=20, top=10)
        BB(left=1, bottom=5, right=20, top=10)

    Or partially, for example like this:
        >>> BB(right=5, top=10)
        BB(left=0, bottom=0, right=5, top=10)
    """

    left: float = 0
    bottom: float = 0
    right: float = 0
    top: float = 0

    @property
    def height(self) -> float:
        """Height of bounding box."""
        return self.top - self.bottom

    @property
    def width(self) -> float:
        """Width of bounding box."""
        return self.right - self.left

    @property
    def position(self) -> Vec2d:
        """Position of the center. Alias to self.center()"""
        return self.center()

    @staticmethod
    def newForCircle(p: VecLike, r: float) -> "BB":
        """Convenience constructor for making a BB fitting a circle at
        position p with radius r.
        """

        bb_ = lib.cpBBNewForCircle(p, r)
        return BB(bb_.l, bb_.b, bb_.r, bb_.t)

    # Called for `obj in BB`.
    def __contains__(self, item):
        if isinstance(item, BB):
            return self.contains(item)
        elif isinstance(item, (tuple, Vec2d)):
            return self.contains_vect(item)
        # TODO include other shape queries.
        else:
            return False

    def intersects(self, other: "BB") -> bool:
        """Returns true if the bounding boxes intersect"""
        return bool(lib.cpBBIntersects(self, other))

    def intersects_segment(self, a: VecLike, b: VecLike) -> bool:
        """Returns true if the segment defined by endpoints a and b
        intersect this bb."""
        assert len(a) == 2
        assert len(b) == 2
        return bool(lib.cpBBIntersectsSegment(self, a, b))

    def contains(self, other: "BB") -> bool:
        """Returns true if bb completely contains the other bb"""
        return bool(lib.cpBBContainsBB(self, other))

    def contains_vect(self, v: VecLike) -> bool:
        """Returns true if this bb contains the vector v"""
        assert len(v) == 2
        return bool(lib.cpBBContainsVect(self, v))

    def merge(self, other: "BB") -> "BB":
        """Return the minimal bounding box that contains both this bb and the
        other bb
        """
        cp_bb = lib.cpBBMerge(self, other)
        return BB(cp_bb.l, cp_bb.b, cp_bb.r, cp_bb.t)

    def expand(self, v: VecLike) -> "BB":
        """Return the minimal bounding box that contans both this bounding box
        and the vector v
        """
        cp_bb = lib.cpBBExpand(self, tuple(v))
        return BB(cp_bb.l, cp_bb.b, cp_bb.r, cp_bb.t)

    def translate(self, delta: VecLike) -> "BB":
        """
        Displace BB to the given displacement vector.
        """
        dx, dy = delta
        return BB(self.left + dx, self.bottom + dy, self.right + dx, self.top + dy)

    def center(self) -> Vec2d:
        """Return the center"""
        v = lib.cpBBCenter(self)
        return Vec2d(v.x, v.y)

    def vertices(self) -> Iterable[Vec2d]:
        """Iterate over all vertices of the bounding box.

        Cycle counter-clockwise from bottom-left.
        """
        yield Vec2d(self.left, self.bottom)
        yield Vec2d(self.right, self.bottom)
        yield Vec2d(self.right, self.top)
        yield Vec2d(self.left, self.top)

    def area(self) -> float:
        """Return the area"""
        return lib.cpBBArea(self)

    def merged_area(self, other: "BB") -> float:
        """Merges this and other then returns the area of the merged bounding
        box.
        """
        return lib.cpBBMergedArea(self, other)

    def segment_query(self, a: VecLike, b: VecLike) -> float:
        """Returns the fraction along the segment query the BB is hit.

        Returns infinity if it doesnt hit
        """
        assert len(a) == 2
        assert len(b) == 2
        return lib.cpBBSegmentQuery(self, a, b)

    def clamp_vect(self, v: VecLike) -> Vec2d:
        """Returns a copy of the vector v clamped to the bounding box"""
        assert len(v) == 2
        v2 = lib.cpBBClampVect(self, v)
        return Vec2d(v2.x, v2.y)

    def wrap_vect(self, v: VecLike) -> Vec2d:
        """Returns a copy of v wrapped to the bounding box."""
        return lib._cpBBWrapVect(self, v)
