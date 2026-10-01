"""
rl - Reinforcement Learning
Copyright (C) 2016  Joey Andres<yeojserdna@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
)

try:
    from rl.agent.StateAction import StateAction
    from rl.agent.StateActionNotExistException import StateActionNotExistException
except ImportError:
    try:
        from .StateAction import StateAction
        from .StateActionNotExistException import StateActionNotExistException
    except ImportError:
        try:
            from StateAction import StateAction
            from StateActionNotExistException import StateActionNotExistException
        except ImportError:
            class StateActionNotExistException(Exception):
                """Handling situations when StateAction being queried does not exist."""

                def __init__(self, extra_message: str = "") -> None:
                    super().__init__(extra_message)
                    self._extra_message: str = extra_message

                def what(self) -> str:
                    return self._extra_message

                def __str__(self) -> str:
                    return self._extra_message

            class StateAction(Generic[TypeVar("S"), TypeVar("A")]):
                """Encapsulates state action pair."""

                def __init__(self, state: Any = None, action: Any = None) -> None:
                    self._state = state
                    self._action = action

                def getState(self) -> Any:
                    return self._state

                def getAction(self) -> Any:
                    return self._action

                def setState(self, state: Any) -> None:
                    self._state = state

                def setAction(self, action: Any) -> None:
                    self._action = action

                def __eq__(self, other: Any) -> bool:
                    if not isinstance(other, StateAction):
                        return False
                    return self._state == other._state and self._action == other._action

                def __hash__(self) -> int:
                    return hash((self._state, self._action))

                def __repr__(self) -> str:
                    return f"StateAction({self._state!r}, {self._action!r})"

S = TypeVar("S")
A = TypeVar("A")


class StateActionPairContainer(Generic[S, A]):
    """Encapsulates the mapping of StateAction to their Value.

    Args:
        S: State data type.
        A: Action data type.
    """

    def __init__(
        self,
        sa_map: Optional[MutableMapping[StateAction[S, A], float]] = None,
    ) -> None:
        """No-arg constructor or initialize with custom mapping."""
        if sa_map is not None:
            self._stateActionPairMap: MutableMapping[StateAction[S, A], float] = sa_map
        else:
            self._stateActionPairMap: MutableMapping[StateAction[S, A], float] = {}

    def addState(
        self,
        state: S,
        value: float,
        actionSet: Iterable[A],
    ) -> None:
        """Adds a new state with the corresponding actions.

        Uses insert semantics (does not overwrite existing pairs).
        """
        for action in actionSet:
            pair = StateAction(state, action)
            if pair not in self._stateActionPairMap:
                self._stateActionPairMap[pair] = float(value)

    add_state = addState

    def addStateAction(
        self,
        stateAction: StateAction[S, A],
        value: float,
    ) -> None:
        """Adds a new state with the corresponding action.

        Uses insert semantics (does not overwrite existing pairs).
        """
        if stateAction not in self._stateActionPairMap:
            self._stateActionPairMap[stateAction] = float(value)

    add_state_action = addStateAction

    def addAction(
        self,
        action: A,
        value: float,
        stateSet: Iterable[S],
    ) -> None:
        """Adds a new action across a set of states.

        Uses insert semantics (does not overwrite existing pairs).
        """
        for state in stateSet:
            pair = StateAction(state, action)
            if pair not in self._stateActionPairMap:
                self._stateActionPairMap[pair] = float(value)

    add_action = addAction

    def stateInStateActionPairMap(
        self,
        state: S,
        actionSet: Iterable[A],
    ) -> bool:
        """Checks whether state is in _stateActionPairMap using a sample action."""
        action_iter = iter(actionSet)
        try:
            sample_action = next(action_iter)
        except StopIteration:
            return False
        return StateAction(state, sample_action) in self._stateActionPairMap

    state_in_state_action_pair_map = stateInStateActionPairMap

    def getStateActionValue(
        self,
        stateAction: StateAction[S, A],
    ) -> float:
        """Returns the value of stateAction.

        Raises:
            StateActionNotExistException: When stateAction does not exist.
        """
        try:
            return self._stateActionPairMap[stateAction]
        except KeyError:
            raise StateActionNotExistException("State-Pair given is not yet added.")

    get_state_action_value = getStateActionValue

    def setStateActionValue(
        self,
        stateAction: StateAction[S, A],
        value: float,
    ) -> None:
        """Sets the value of stateAction."""
        self._stateActionPairMap[stateAction] = float(value)

    set_state_action_value = setStateActionValue

    def operator_brackets(
        self,
        stateAction: StateAction[S, A],
    ) -> float:
        """Access state-action value via [] operator."""
        return self.getStateActionValue(stateAction)

    def __getitem__(
        self,
        stateAction: StateAction[S, A],
    ) -> float:
        return self.getStateActionValue(stateAction)

    def __setitem__(
        self,
        stateAction: StateAction[S, A],
        value: float,
    ) -> None:
        self.setStateActionValue(stateAction, value)

    def __contains__(
        self,
        stateAction: StateAction[S, A],
    ) -> bool:
        return stateAction in self._stateActionPairMap

    def begin(self) -> Iterator[Tuple[StateAction[S, A], float]]:
        """Returns begin iterator of state-action to value map."""
        return iter(self._stateActionPairMap.items())

    def end(self) -> None:
        """Returns end iterator of state-action to value map."""
        return None

    def __iter__(self) -> Iterator[StateAction[S, A]]:
        return iter(self._stateActionPairMap)

    def items(self) -> Iterator[Tuple[StateAction[S, A], float]]:
        return iter(self._stateActionPairMap.items())

    def keys(self) -> Iterator[StateAction[S, A]]:
        return iter(self._stateActionPairMap.keys())

    def values(self) -> Iterator[float]:
        return iter(self._stateActionPairMap.values())

    def size(self) -> int:
        """Returns state-action pair count."""
        return len(self._stateActionPairMap)

    def __len__(self) -> int:
        return len(self._stateActionPairMap)


StateActionPairContainerSimple = StateActionPairContainer