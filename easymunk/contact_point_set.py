import dataclasses
from typing import List

from .vec2d import Vec2d


@dataclasses.dataclass(frozen=True)
class ContactPoint:
    """Contains information about a contact point.

    point_a and point_b are the contact position on the surface of each shape.

    distance is the penetration distance of the two shapes. Overlapping
    means it will be negative. This value is calculated as
    dot(point2 - point1), normal) and is ignored when you set the
    Arbiter.contact_point_set.
    """

    point_a: Vec2d
    point_b: Vec2d
    distance: float


@dataclasses.dataclass(frozen=True)
class ContactPointSet:
    """Contact point sets make getting contact information simpler.

    normal is the normal of the collision

    points is the array of contact points. Can be at most 2 points.
    """

    normal: Vec2d
    points: List[ContactPoint]


def contact_point_set_from_cffi(cp_point_set) -> ContactPointSet:
    normal = Vec2d(cp_point_set.normal.x, cp_point_set.normal.y)

    points = []
    for i in range(cp_point_set.count):
        ps = cp_point_set.points[i]
        points.append(ContactPoint(
            Vec2d(ps.pointA.x, ps.pointA.y),
            Vec2d(ps.pointB.x, ps.pointB.y),
            ps.distance,
        ))

    return ContactPointSet(normal, points)
