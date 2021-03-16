from abc import ABC
from functools import reduce
from typing import (
    Any,
    Tuple,
    TypeVar,
    Iterable,
    TYPE_CHECKING,
    Set,
)


if TYPE_CHECKING:
    from .bb import BB

T = TypeVar("T", bound="PickleMixin")


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
