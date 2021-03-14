import operator
from abc import abstractmethod, ABC
from functools import reduce
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    TypeVar,
    Iterable,
    Callable,
    TYPE_CHECKING,
    Iterator,
    Set,
)

from ._chipmunk_cffi import lib
from .util import compose_filters, single_query

if TYPE_CHECKING:
    from .body import Body
    from .shapes import Shape
    from .constraints import Constraint
    from .bb import BB

T = TypeVar("T", bound="PickleMixin")
Predicate = Callable[..., bool]
_State = Dict[str, List[Tuple[str, Any]]]

BODY_TYPE_MAP = {
    "dynamic": lib.CP_BODY_TYPE_DYNAMIC,
    "kinematic": lib.CP_BODY_TYPE_KINEMATIC,
    "static": lib.CP_BODY_TYPE_STATIC,
}


class PickleMixin(ABC):
    """PickleMixin is used to provide base functionality for pickle/unpickle
    and copy.
    """

    _pickle_args: Tuple[str]
    _pickle_kwargs: Tuple[str]
    _pickle_meta_hide: Set[str]

    def __getstate__(self):
        args = [getattr(self, k) for k in self._pickle_args]
        meta = dict(self.__dict__)
        for k in self._pickle_meta_hide:
            meta.pop(k, None)
        for k in self._pickle_kwargs:
            meta[k] = getattr(self, k)
        return args, meta

    def __setstate__(self, state):
        args, meta = state
        # noinspection PyArgumentList
        self.__init__(*args)  # type: ignore
        for k, v in meta.items():
            setattr(self, k, v)

    def copy(self: T) -> T:
        """Create a deep copy of this object."""

        state = self.__getstate__()
        new = object.__new__(type(self))
        new.__setstate__(state)
        return new


class TypingAttrMixing:
    """Type helper mixin to make mypy accept dynamic attributes."""

    def __setattr__(self, name: str, value: Any) -> None:
        """Override default setattr to make sure type checking works."""
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        """Override default getattr to make sure type checking works."""
        return self.__getattribute__(name)


class HasBBMixin:
    """Declare interface for elements that have a bounding box and the corresponding
    left, right, top and bottom attributes."""

    def _iter_bounding_boxes(self) -> Iterable["BB"]:
        raise NotImplementedError

    @property
    def bb(self) -> "BB":
        """
        Bounding box for all colliding body shapes.
        """
        merge = lambda a, b: a.merge(b)
        return reduce(merge, self._iter_bounding_boxes())

    @property
    def left(self) -> float:
        """
        Right position (world coordinates) of body.

        Exclude sensor shapes.
        """
        return max(bb.left for bb in self._iter_bounding_boxes())

    @property
    def right(self) -> float:
        """
        Right position (world coordinates) of body.

        Exclude sensor shapes.
        """
        return max(bb.right for bb in self._iter_bounding_boxes())

    @property
    def bottom(self) -> float:
        """
        Bottom position (world coordinates) of body.

        Exclude sensor shapes.
        """
        return max(bb.bottom for bb in self._iter_bounding_boxes())

    @property
    def top(self) -> float:
        """
        Top position (world coordinates) of body.

        Exclude sensor shapes.
        """
        return max(bb.top for bb in self._iter_bounding_boxes())


class FilterElementsMixin:
    """Interface for elements that can filter its children."""

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

    @abstractmethod
    def _iter_bodies(self) -> Iterator["Body"]:
        raise NotImplementedError

    @abstractmethod
    def _iter_shapes(self) -> Iterator["Shape"]:
        raise NotImplementedError

    @abstractmethod
    def _iter_constraints(self) -> Iterator["Constraint"]:
        raise NotImplementedError

    def _generic_filters(self, kwargs) -> Iterable[Predicate]:
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

    def get_body(self, *, first: bool = False, **kwargs) -> "Body":
        """
        Get single body from filter query. Raise ValueError if no body or if
        multiple bodies are found.

        This function accepts the same arguments as filter_bodies().
        If first=True, return the first match, even if multiple matches are
        found.
        """
        return single_query(self.filter_bodies(**kwargs), name="body", first=first)

    def filter_bodies(self, *, body_type=None, **kwargs) -> Iterable["Body"]:
        """
        Filter bodies according to criteria.

        Args:
            body_type: Either "dynamic", "kinematic" or "static". Select only
                bodies of the specified type.
        """
        filters = []
        if body_type:
            body_type = BODY_TYPE_MAP.get(body_type)
            filters.append(lambda b: b.body_type == body_type)
        filters.extend(self._generic_filters(kwargs))
        return filter(compose_filters(filters), self._iter_bodies())

    def get_shape(self, *, first: bool = False, **kwargs) -> "Shape":
        """
        Get single shape from filter query. Raise ValueError if no shape or if
        multiple shapes are found.

        This function accepts the same arguments as filter_shapes().
        If first=True, return the first match, even if multiple matches are
        found.
        """
        return single_query(self.filter_shapes(**kwargs), name="shape", first=first)

    def filter_shapes(self, **kwargs) -> Iterable["Shape"]:
        """
        Filter shapes according to criteria.
        """
        filters = self._generic_filters(kwargs)
        return filter(compose_filters(filters), self._iter_shapes())

    def get_constraint(self, *, first: bool = False, **kwargs) -> "Constraint":
        """
        Get single constraint from filter query. Raise ValueError if no
        constraint or multiple constraints are found.

        This function accepts the same arguments as filter_constraints().
        If first=True, return the first match, even if multiple matches are
        found.
        """
        return single_query(
            self.filter_constraints(**kwargs), name="constraint", first=first
        )

    def filter_constraints(self, **kwargs) -> Iterable["Constraint"]:
        """
        Filter constraints according to criteria.
        """
        filters = self._generic_filters(kwargs)
        return filter(compose_filters(filters), self._iter_constraints())
