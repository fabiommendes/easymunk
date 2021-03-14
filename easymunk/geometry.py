from typing import Sequence, Tuple, cast

from . import _chipmunk_cffi

cp = _chipmunk_cffi.lib
ffi = _chipmunk_cffi.ffi


def moment_for_circle(
    mass: float,
    inner_radius: float,
    outer_radius: float,
    offset: Tuple[float, float] = (0, 0),
) -> float:
    """Calculate the moment of inertia for a hollow circle

    (A solid circle has an inner radius of 0)
    """
    assert len(offset) == 2

    return cp.cpMomentForCircle(mass, inner_radius, outer_radius, offset)


def moment_for_segment(
    mass: float, a: Tuple[float, float], b: Tuple[float, float], radius: float
) -> float:
    """Calculate the moment of inertia for a line segment

    The endpoints a and b are relative to the body
    """
    assert len(a) == 2
    assert len(b) == 2

    return cp.cpMomentForSegment(mass, a, b, radius)


def moment_for_box(mass: float, size: Tuple[float, float]) -> float:
    """Calculate the moment of inertia for a solid box centered on the body.

    size should be a tuple of (width, height)
    """
    assert len(size) == 2
    return cp.cpMomentForBox(mass, size[0], size[1])


def moment_for_poly(
    mass: float,
    vertices: Sequence[Tuple[float, float]],
    offset: Tuple[float, float] = (0, 0),
    radius: float = 0,
) -> float:
    """Calculate the moment of inertia for a solid polygon shape.

    Assumes the polygon center of gravity is at its centroid. The offset is
    added to each vertex.
    """
    assert len(offset) == 2
    vs = list(vertices)
    return cp.cpMomentForPoly(mass, len(vs), vs, offset, radius)


def area_for_circle(inner_radius: float, outer_radius: float) -> float:
    """Area of a hollow circle."""
    return cast(float, cp.cpAreaForCircle(inner_radius, outer_radius))


def area_for_segment(
    a: Tuple[float, float], b: Tuple[float, float], radius: float
) -> float:
    """Area of a beveled segment.

    (Will always be zero if radius is zero)
    """
    assert len(a) == 2
    assert len(b) == 2

    return cp.cpAreaForSegment(a, b, radius)


def area_for_poly(vertices: Sequence[Tuple[float, float]], radius: float = 0) -> float:
    """Signed area of a polygon shape.

    Returns a negative number for polygons with a clockwise winding.
    """
    vs = list(vertices)
    return cp.cpAreaForPoly(len(vs), vs, radius)
