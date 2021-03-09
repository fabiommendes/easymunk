__docformat__ = "reStructuredText"

from typing import TYPE_CHECKING, Tuple

from ._chipmunk_cffi import ffi, lib
from .contact_point_set import ContactPointSet, contact_point_set_from_cffi
from .util import cp_property
from .vec2d import Vec2d, vec2d_from_cffi

if TYPE_CHECKING:
    from .space import Space
    from .shapes import Shape


class Arbiter(object):
    """The Arbiter object encapsulates a pair of colliding shapes and all of
    the data about their collision.

    They are created when a collision starts, and persist until those
    shapes are no longer colliding.

    Warning:
        Because arbiters are handled by the space you should never
        hold onto a reference to an arbiter as you don't know when it will be
        destroyed! Use them within the callback where they are given to you
        and then forget about them or copy out the information you need from
        them.
    """

    def _get_contact_point_set(self) -> ContactPointSet:
        points = lib.cpArbiterGetContactPointSet(self._cffi_ref)
        return contact_point_set_from_cffi(points)

    def _set_contact_point_set(self, point_set: ContactPointSet) -> None:
        # This has to be done by fetching a new Chipmunk point set, update it
        # according to whats passed in and the pass that back to chipmunk due
        # to the fact that ContactPointSet doesnt contain a reference to the
        # corresponding c struct.
        cp_set = lib.cpArbiterGetContactPointSet(self._cffi_ref)
        cp_set.normal = point_set.normal

        if len(point_set.points) == cp_set.count:
            points = cp_set.points
            for i in range(cp_set.count):
                points[i].pointA = point_set.points[i].point_a
                points[i].pointB = point_set.points[i].point_b
                points[i].distance = point_set.points[i].distance
        else:
            msg = "Expected {} points, got {} points in point_set".format(
                cp_set.count, len(point_set.points)
            )
            raise Exception(msg)

        lib.cpArbiterSetContactPointSet(self._cffi_ref, ffi.addressof(cp_set))

    contact_point_set: ContactPointSet = property(
        _get_contact_point_set,
        _set_contact_point_set,
        doc="""Contact point sets make getting contact information from the 
        Arbiter simpler.
        
        Return `ContactPointSet`""",
    )
    restitution: float = cp_property(
        "Arbiter",
        "Restitution",
        doc="""The calculated restitution (elasticity) for this collision 
        pair. 
        
        Setting the value in a pre_solve() callback will override the value 
        calculated by the space. The default calculation multiplies the 
        elasticity of the two shapes together.
        """,
    )
    friction: float = cp_property(
        "Arbiter",
        "Friction",
        doc="""The calculated friction for this collision pair. 
        
        Setting the value in a pre_solve() callback will override the value 
        calculated by the space. The default calculation multiplies the 
        friction of the two shapes together.
        """,
    )
    surface_velocity: Vec2d = cp_property(
        "Arbiter",
        "SurfaceVelocity",
        doc="""The calculated surface velocity for this collision pair. 
        
        Setting the value in a pre_solve() callback will override the value 
        calculated by the space. the default calculation subtracts the 
        surface velocity of the second shape from the first and then projects 
        that onto the tangent of the collision. This is so that only 
        friction is affected by default calculation. Using a custom 
        calculation, you can make something that responds like a pinball 
        bumper, or where the surface velocity is dependent on the location 
        of the contact point.
        """,
        wrap=vec2d_from_cffi,
    )

    @property
    def total_impulse(self) -> Vec2d:
        """Returns the impulse that was applied this step to resolve the
        collision.

        This property should only be called from a post-solve or each_arbiter
        callback.
        """
        v = lib.cpArbiterTotalImpulse(self._cffi_ref)
        return Vec2d(v.x, v.y)

    @property
    def total_ke(self) -> float:
        """The amount of energy lost in a collision including static, but
        not dynamic friction.

        This property should only be called from a post-solve or each_arbiter callback.
        """
        return lib.cpArbiterTotalKE(self._cffi_ref)

    @property
    def is_first_contact(self) -> bool:
        """Returns true if this is the first step the two shapes started
        touching.

        This can be useful for sound effects for instance. If its the first
        frame for a certain collision, check the energy of the collision in a
        post_step() callback and use that to determine the volume of a sound
        effect to play.
        """
        return bool(lib.cpArbiterIsFirstContact(self._cffi_ref))

    @property
    def is_removal(self) -> bool:
        """Returns True during a separate() callback if the callback was
        invoked due to an object removal.
        """
        return bool(lib.cpArbiterIsRemoval(self._cffi_ref))

    @property
    def normal(self) -> Vec2d:
        """Returns the normal of the collision."""
        v = lib.cpArbiterGetNormal(self._cffi_ref)
        return Vec2d(v.x, v.y)

    @property
    def shapes(self) -> Tuple["Shape", "Shape"]:
        """Get the shapes in the order that they were defined in the
        collision handler associated with this arbiter
        """
        ptr_a = ffi.new("cpShape *[1]")
        ptr_b = ffi.new("cpShape *[1]")

        lib.cpArbiterGetShapes(self._cffi_ref, ptr_a, ptr_b)
        a = self._space._shape_from_cffi(ptr_a[0])
        b = self._space._shape_from_cffi(ptr_b[0])
        if a is None or b is None:
            raise ValueError('invalid shape in arbiter')
        return a, b

    def __init__(self, _arbiter: ffi.CData, space: "Space") -> None:
        """Initialize an Arbiter object from the Chipmunk equivalent struct
        and the Space.

        .. note::
            You should never need to create an instance of this class directly.
        """

        self._cffi_ref = _arbiter
        self._space = space
