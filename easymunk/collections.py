import operator
from typing import (
    Sequence as Set,
    TypeVar,
    Dict,
    Callable,
    Any,
    Iterable,
    List,
    TYPE_CHECKING,
)
from itertools import chain
import sidekick as sk
from .util import single_query, compose_filters
from . import _chipmunk_cffi

if TYPE_CHECKING:
    from .body import Body

T = TypeVar("T")
Predicate = Callable[..., bool]
cp = _chipmunk_cffi.lib
ffi = _chipmunk_cffi.ffi

BODY_TYPES = {
    "dynamic": cp.CP_BODY_TYPE_DYNAMIC,
    "kinematic": cp.CP_BODY_TYPE_KINEMATIC,
    "static": cp.CP_BODY_TYPE_STATIC,
}


class Objects(Set[T]):
    """
    Collections of objects with richer APIs.

    Base class for the .shapes, .bodies, .constraints accessors.
    """

    _TAIL_MODIFIERS: Dict[str, Callable[[Any, Any], bool]] = {
        "gt": operator.gt,
        "ge": operator.ge,
        "lt": operator.lt,
        "le": operator.le,
        "eq": operator.eq,
        "ne": operator.ne,
        "len": lambda obj, n: len(obj) == n,
    }
    _CHAIN_MODIFIERS: Dict[str, Callable[[Any], Any]] = {
        "len": len,
    }

    def __init__(self, owner, objects):
        self.owner = owner
        self._objects = objects

    def __repr__(self):
        return f"{type(self).__name__}({self._as_list()})"

    def __iter__(self):
        return iter(self._objects)

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

    def _generic_filter_map(self, kwargs) -> Iterable[Predicate]:
        for k, v in kwargs.items():
            name, _, mods = k.split("__")
            yield self._generic_filter(name, v, mods)

    # noinspection PyUnresolvedReferences
    def _generic_filter(self, name: str, value: Any, modifiers: List[str]) -> Predicate:
        """A Django-like filter query.

        It understands a simple DSL like in the example:

        >>> space.filter_objects(mass__gt=10)  # doctest: +SKIP
        ...
        """

        if not modifiers:
            return lambda o: getattr(o, name) == value

        if len(modifiers) == 1:
            mod_fn = self._tail_modifier(modifiers[0])
            return lambda o: mod_fn(o, value)

        *chain, tail = modifiers
        chain_fns = [*map(self._chain_modifier, chain)]
        tail_fn = self._tail_modifier(tail)

        def filter_with_modifiers(o):
            for fn in chain_fns:
                o = fn(o)
            return tail_fn(o, value)

        return filter_with_modifiers

    def _tail_modifier(self, name) -> Callable[[Any, Any], bool]:
        try:
            return self._TAIL_MODIFIERS[name]
        except KeyError:
            raise ValueError(f"invalid modifier: {name}")

    def _chain_modifier(self, name) -> Callable[[Any], Any]:
        try:
            return self._CHAIN_MODIFIERS[name]
        except KeyError:
            return operator.attrgetter(name)

    def get(self, *, first: bool = False, **kwargs) -> T:
        """
        Get single element from filter query. Raise ValueError if no element or
        if multiple values are found.

        This function accepts the same arguments as filter().
        If first=True, return the first match, even if multiple matches are
        found.
        """
        return single_query(self.filter(**kwargs), name="body", first=first)

    def filter(self, **kwargs) -> Iterable[T]:
        """
        Filter elements according to criteria.

        Args:
            body_type: Either "dynamic", "kinematic" or "static". Select only
                bodies of the specified type.
        """
        filters = []
        filters.extend(self._generic_filter_map(kwargs))
        return sk.filter(compose_filters(filters), self._objects)


class Shapes(Objects["Shape"]):
    """
    Collection of shapes.
    """


class Bodies(Objects["Body"]):
    """
    Collection of bodies.
    """

    def _generic_filter_map(self, kwargs):
        filters = super()._generic_filter_map(kwargs)
        body_type = kwargs.pop("body_type", None)
        if body_type is not None:
            body_type = BODY_TYPES.get(body_type, body_type)
            return chain(filters, lambda b: b.body_type == body_type)
        return filters


class Constraints(Objects["Constraint"]):
    """
    Collection of constraints.
    """
