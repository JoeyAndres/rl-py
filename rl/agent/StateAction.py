from __future__ import annotations

from typing import Generic, Optional, TypeVar, Union, overload

S = TypeVar("S")
A = TypeVar("A")


class StateAction(Generic[S, A]):
    """Encapsulates state action pair.

    Requires both S state and A action data type to be comparable.
    """

    _state: Optional[S]
    _action: Optional[A]

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, state: StateAction[S, A]) -> None: ...

    @overload
    def __init__(self, state: S, action: A) -> None: ...

    def __init__(
        self,
        state: Optional[Union[S, StateAction[S, A]]] = None,
        action: Optional[A] = None,
    ) -> None:
        """Constructor for state-action pair, copy constructor, or no-arg constructor."""
        if isinstance(state, StateAction) and action is None:
            self._state = state._state
            self._action = state._action
        else:
            self._state = state  # type: ignore[assignment]
            self._action = action

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, StateAction):
            return NotImplemented
        if self._state < other._state:
            return True
        if self._state > other._state:
            return False
        if self._action < other._action:
            return True
        return False

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, StateAction):
            return NotImplemented
        if self._state > other._state:
            return True
        if self._state < other._state:
            return False
        if self._action > other._action:
            return True
        return False

    def __le__(self, other: object) -> bool:
        if not isinstance(other, StateAction):
            return NotImplemented
        return not (self > other)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, StateAction):
            return NotImplemented
        return not (self < other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateAction):
            return NotImplemented
        return self._state == other._state and self._action == other._action

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, StateAction):
            return NotImplemented
        return not (self == other)

    def __hash__(self) -> int:
        return hash((self._state, self._action))

    def __repr__(self) -> str:
        return f"StateAction(state={self._state!r}, action={self._action!r})"

    def getState(self) -> Optional[S]:
        """Return state of state-action pair."""
        return self._state

    def getAction(self) -> Optional[A]:
        """Return action of state-action pair."""
        return self._action

    def setState(self, state: Optional[S]) -> None:
        """Set the state of state-action pair."""
        self._state = state

    def setAction(self, action: Optional[A]) -> None:
        """Set the action of state-action pair."""
        self._action = action

    @property
    def state(self) -> Optional[S]:
        return self._state

    @state.setter
    def state(self, state: Optional[S]) -> None:
        self._state = state

    @property
    def action(self) -> Optional[A]:
        return self._action

    @action.setter
    def action(self, action: Optional[A]) -> None:
        self._action = action