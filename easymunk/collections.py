from typing import Sequence as Set, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .shapes import Shape
    from .body import Body
    from .constraints import Constraint

T = TypeVar("T")


class Objects(Set[T]):
    """
    Collections of objects with richer APIs.

    Base class for the .shapes, .bodies, .constraints accessors.
    """

    def __init__(self, owner, objects):
        self.owner = owner
        self._objects = objects

    def __repr__(self):
        return f"{type(self).__name__}({self._as_list()})"

    def __len__(self) -> int:
        return len(self._objects)

    def __contains__(self, item):
        return item in self._objects

    def __getitem__(self, i: int) -> T:
        for j, obj in enumerate(self._objects):
            if i == j:
                return obj
        raise IndexError(i)

    def __eq__(self, other):
        if isinstance(other, list):
            return self._as_list() == other
        elif isinstance(other, set):
            return set(self._objects) == other
        elif isinstance(other, Objects):
            return set(self._objects) == set(other._objects)
        return NotImplemented

    def _as_list(self):
        return list(self._objects)


class Shapes(Objects["Shape"]):
    """
    Collection of shapes.
    """


class Bodies(Objects["Body"]):
    """
    Collection of bodies.
    """


class Constraints(Objects["Constraint"]):
    """
    Collection of constraints.
    """
