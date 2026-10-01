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

import math
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    final,
)

S = TypeVar("S")
A = TypeVar("A")

try:
    from rl.agent.StateAction import StateAction
    from rl.agent.StateActionTransition import StateActionTransition
    from rl.policy.Policy import Policy
    from rl.algorithm.DynaQ import DynaQ
except ImportError:
    try:
        from ..agent.StateAction import StateAction
        from ..agent.StateActionTransition import StateActionTransition
        from ..policy.Policy import Policy
        from .DynaQ import DynaQ
    except (ImportError, ValueError):
        class StateAction(Generic[S, A]):
            def __init__(self, state: S, action: A) -> None:
                self._state: S = state
                self._action: A = action

            def getState(self) -> S:
                return self._state

            def getAction(self) -> A:
                return self._action

            def setState(self, state: S) -> None:
                self._state = state

            def setAction(self, action: A) -> None:
                self._action = action

            def __lt__(self, other: Any) -> bool:
                if not isinstance(other, StateAction):
                    return NotImplemented
                return (self._state, self._action) < (other._state, other._action)

            def __gt__(self, other: Any) -> bool:
                if not isinstance(other, StateAction):
                    return NotImplemented
                return (self._state, self._action) > (other._state, other._action)

            def __eq__(self, other: Any) -> bool:
                if not isinstance(other, StateAction):
                    return False
                return (self._state, self._action) == (other._state, other._action)

            def __hash__(self) -> int:
                return hash((self._state, self._action))

        class StateActionTransition(Generic[S]):
            def __init__(self, greediness: float = 1.0, stepSize: float = 0.1) -> None:
                self._greediness = greediness
                self._stepSize = stepSize

            def update(self, nextState: S, reward: float) -> None:
                pass

            def getReward(self, state: S) -> float:
                return 0.0

            def getNextState(self) -> S:
                raise NotImplementedError

        class Policy(Generic[S, A]):
            pass

        class DynaQ(Generic[S, A]):
            def __init__(
                self,
                stepSize: float = 0.0,
                discountRate: float = 0.0,
                policy: Any = None,
                simulationIterationCount: int = 0,
                stateTransitionGreediness: float = 0.0,
                stateTransitionStepSize: float = 0.0,
            ) -> None:
                self._stepSize: float = stepSize
                self._discountRate: float = discountRate
                self._policy: Any = policy
                self._simulationIterationCount: int = simulationIterationCount
                self._stateTransitionGreediness: float = stateTransitionGreediness
                self._stateTransitionStepSize: float = stateTransitionStepSize
                self._model: Dict[StateAction[S, A], StateActionTransition[S]] = {}
                self._stateActionPairContainer: Dict[StateAction[S, A], float] = {}

            def update(
                self,
                currentStateAction: StateAction[S, A],
                nextState: S,
                reward: float,
                actionSet: Union[Set[A], List[A]],
            ) -> None:
                pass

            def _updateModel(
                self,
                currentStateAction: StateAction[S, A],
                nextState: S,
                reward: float,
            ) -> None:
                pass

            def argMax(self, state: S, actionSet: Union[Set[A], List[A]]) -> A:
                raise NotImplementedError

            def getLearningAction(self, state: S, actionSet: Union[Set[A], List[A]]) -> A:
                raise NotImplementedError

            def backUpStateActionPair(
                self,
                currentStateAction: StateAction[S, A],
                reward: float,
                nextStateAction: StateAction[S, A],
            ) -> None:
                pass


class StateActionPair(NamedTuple, Generic[S, A]):
    first: StateAction[S, A]
    second: float


class StateActionPairValueComparison(Generic[S, A]):
    """Comparison object for StateActionPair used in priority queue."""

    def __call__(
        self,
        lhs: Union[Tuple[StateAction[S, A], float], StateActionPair[S, A]],
        rhs: Union[Tuple[StateAction[S, A], float], StateActionPair[S, A]],
    ) -> bool:
        lhs_sa, lhs_val = lhs[0], lhs[1]
        rhs_sa, rhs_val = rhs[0], rhs[1]
        try:
            if lhs_sa < rhs_sa:
                return True
            elif lhs_sa > rhs_sa:
                return False
        except TypeError:
            pass
        return lhs_val < rhs_val


class PriorityQueue(Generic[S, A]):
    """Max-heap priority queue matching std::priority_queue with StateActionPairValueComparison."""

    def __init__(
        self,
        comparator: Optional[Callable[[Any, Any], bool]] = None,
    ) -> None:
        self._heap: List[StateActionPair[S, A]] = []
        self._comparator = comparator or StateActionPairValueComparison[S, A]()

    def _comp(self, a: StateActionPair[S, A], b: StateActionPair[S, A]) -> bool:
        return bool(self._comparator(a, b))

    def empty(self) -> bool:
        return len(self._heap) == 0

    def is_empty(self) -> bool:
        return self.empty()

    def size(self) -> int:
        return len(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return not self.empty()

    def top(self) -> StateActionPair[S, A]:
        if not self._heap:
            raise IndexError("top from empty priority queue")
        return self._heap[0]

    def peek(self) -> StateActionPair[S, A]:
        return self.top()

    def push(
        self,
        item: Union[StateActionPair[S, A], Tuple[StateAction[S, A], float]],
    ) -> None:
        if not isinstance(item, StateActionPair):
            item = StateActionPair(item[0], item[1])
        self._heap.append(item)
        self._sift_up(len(self._heap) - 1)

    def pop(self) -> StateActionPair[S, A]:
        if not self._heap:
            raise IndexError("pop from empty priority queue")
        max_item = self._heap[0]
        last_item = self._heap.pop()
        if self._heap:
            self._heap[0] = last_item
            self._sift_down(0)
        return max_item

    def _sift_up(self, idx: int) -> None:
        child_idx = idx
        while child_idx > 0:
            parent_idx = (child_idx - 1) // 2
            if self._comp(self._heap[parent_idx], self._heap[child_idx]):
                self._heap[parent_idx], self._heap[child_idx] = (
                    self._heap[child_idx],
                    self._heap[parent_idx],
                )
                child_idx = parent_idx
            else:
                break

    def _sift_down(self, idx: int) -> None:
        heap_len = len(self._heap)
        parent_idx = idx
        while True:
            left_child = 2 * parent_idx + 1
            right_child = 2 * parent_idx + 2
            largest = parent_idx

            if left_child < heap_len and self._comp(self._heap[largest], self._heap[left_child]):
                largest = left_child

            if right_child < heap_len and self._comp(self._heap[largest], self._heap[right_child]):
                largest = right_child

            if largest != parent_idx:
                self._heap[parent_idx], self._heap[largest] = (
                    self._heap[largest],
                    self._heap[parent_idx],
                )
                parent_idx = largest
            else:
                break


@final
class DynaQPrioritizedSweeping(DynaQ[S, A], Generic[S, A]):
    """DynaQ Prioritize sweeping implementation.

    Prioritize Sweeping only backs up states that factor the most to
    achieving the goal during simulation. This way convergence is achieved
    faster. Very useful in large state spaces.
    """

    StateActionPair = StateActionPair

    def __init__(
        self,
        stepSize: float,
        discountRate: float,
        policy: Any,
        simulationIterationCount: int,
        stateTransitionGreediness: float,
        stateTransitionStepSize: float,
        priorityThreshold: float,
    ) -> None:
        """
        :param stepSize: range [0.0, 1.0]. High step size means faster learning, but
            less precise convergence.
        :param discountRate: range [0.0, 1.0]. High discount rate means more
            consideration of future events.
        :param policy: online policy, that is policy used for action selection.
        :param simulationIterationCount: number of simulations per update/backup.
        :param stateTransitionGreediness: greediness in selecting highest value model.
        :param stateTransitionStepSize: how fast does a model update a value of a
            state-action pair.
        :param priorityThreshold: range [0.0, 1.0], a low value means don't sweep
            too far back. A high priority means sweep far back.
        """
        super().__init__(
            stepSize,
            discountRate,
            policy,
            simulationIterationCount,
            stateTransitionGreediness,
            stateTransitionStepSize,
        )
        self._priorityThreshold: float = priorityThreshold
        self._priorityQueue: PriorityQueue[S, A] = PriorityQueue()

    def update(
        self,
        currentStateAction: StateAction[S, A],
        nextState: S,
        reward: float,
        actionSet: Union[Set[A], List[A]],
    ) -> None:
        if hasattr(self, "argMax"):
            nextAction = self.argMax(nextState, actionSet)
        else:
            nextAction = self.arg_max(nextState, actionSet)

        nextStateAction = StateAction[S, A](nextState, nextAction)

        super().update(currentStateAction, nextState, reward, actionSet)

        # Update model.
        if hasattr(self, "_updateModel"):
            self._updateModel(currentStateAction, nextState, reward)
        elif hasattr(self, "_update_model"):
            self._update_model(currentStateAction, nextState, reward)

        # Acquire A' and tdError.
        tdError = self._getTDError(currentStateAction, reward, nextStateAction)

        # Acquire priority value.
        priority = abs(tdError)
        if priority > self._priorityThreshold:
            self._priorityQueue.push(StateActionPair(currentStateAction, priority))

        # Repeat n times while priority queue is not empty.
        self._prioritySweep(actionSet)

    def _prioritySweep(self, actionSet: Union[Set[A], List[A]]) -> None:
        # Repeat n times while priority queue is not empty.
        for _ in range(self._simulationIterationCount):
            if self._priorityQueue.empty():
                break

            currentStateActionPair: StateAction[S, A] = self._priorityQueue.top().first
            self._priorityQueue.pop()

            # Acquire next reward from model.
            try:
                currentStateActionTransition = self._model[currentStateActionPair]
            except KeyError:
                continue

            if hasattr(currentStateActionTransition, "getNextState"):
                nextState = currentStateActionTransition.getNextState()
            else:
                nextState = currentStateActionTransition.get_next_state()

            if hasattr(currentStateActionTransition, "getReward"):
                nextReward = currentStateActionTransition.getReward(nextState)
            else:
                nextReward = currentStateActionTransition.get_reward(nextState)

            # Acquire max_action(Q(S, a)).
            if hasattr(self, "getLearningAction"):
                nextAction = self.getLearningAction(nextState, actionSet)
            else:
                nextAction = self.get_learning_action(nextState, actionSet)

            nextStateAction = StateAction[S, A](nextState, nextAction)

            # Update Q(S, A).
            if hasattr(self, "backUpStateActionPair"):
                self.backUpStateActionPair(
                    currentStateActionPair, nextReward, nextStateAction
                )
            else:
                self.back_up_state_action_pair(
                    currentStateActionPair, nextReward, nextStateAction
                )

            curr_state = (
                currentStateActionPair.getState()
                if hasattr(currentStateActionPair, "getState")
                else getattr(currentStateActionPair, "state", currentStateActionPair)
            )

            model_items = (
                list(self._model.items())
                if hasattr(self._model, "items")
                else []
            )
            for currentSA, modelStateTransition in model_items:
                # Acquire a model(S) and look for a model(S) that transition to
                # current(S, A).
                if hasattr(modelStateTransition, "getNextState"):
                    modelNextState = modelStateTransition.getNextState()
                else:
                    modelNextState = modelStateTransition.get_next_state()

                # If model(S) -> current(S, A), that means model(S) factors in reaching
                # terminal state.
                # And because of that, back up model(S).
                if modelNextState == curr_state:
                    if hasattr(modelStateTransition, "getReward"):
                        priorReward = modelStateTransition.getReward(modelNextState)
                    else:
                        priorReward = modelStateTransition.get_reward(modelNextState)

                    if hasattr(self, "argMax"):
                        nextModelAction = self.argMax(modelNextState, actionSet)
                    else:
                        nextModelAction = self.arg_max(modelNextState, actionSet)

                    nextSA = StateAction[S, A](modelNextState, nextModelAction)
                    temptdError = self._getTDError(currentSA, priorReward, nextSA)

                    tempPriority = abs(temptdError)
                    if tempPriority > self._priorityThreshold:
                        self._priorityQueue.push(StateActionPair(currentSA, tempPriority))

    def _getTDError(
        self,
        currentStateAction: StateAction[S, A],
        reward: float,
        nextStateAction: StateAction[S, A],
    ) -> float:
        try:
            next_val = self._stateActionPairContainer[nextStateAction]
        except (KeyError, IndexError):
            next_val = 0.0

        try:
            curr_val = self._stateActionPairContainer[currentStateAction]
        except (KeyError, IndexError):
            curr_val = 0.0

        tdError: float = reward + self._discountRate * next_val - curr_val
        return tdError

    _priority_sweep = _prioritySweep
    _get_td_error = _getTDError

    @property
    def priority_threshold(self) -> float:
        return self._priorityThreshold

    @priority_threshold.setter
    def priority_threshold(self, value: float) -> None:
        self._priorityThreshold = value

    @property
    def priority_queue(self) -> PriorityQueue[S, A]:
        return self._priorityQueue