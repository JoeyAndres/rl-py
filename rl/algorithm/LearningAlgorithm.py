"""
rl - Reinforcement Learning
Copyright (C) 2016  Joey Andres <yeojserdna@gmail.com>

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

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Set, TypeVar

S = TypeVar("S")
A = TypeVar("A")

FLOAT = float

try:
    from rl.declares import DEFAULT_GREEDINESS_LEARNING_POLICY
except ImportError:
    try:
        from ..declares import DEFAULT_GREEDINESS_LEARNING_POLICY
    except (ImportError, ValueError):
        DEFAULT_GREEDINESS_LEARNING_POLICY: float = 1.0

try:
    from rl.agent.StateAction import StateAction
except ImportError:
    try:
        from ..agent.StateAction import StateAction
    except (ImportError, ValueError):
        class StateAction(Generic[S, A]):  # type: ignore
            def __init__(self, state: Optional[S] = None, action: Optional[A] = None) -> None:
                self.state = state
                self.action = action

            def getState(self) -> Optional[S]:
                return self.state

            def getAction(self) -> Optional[A]:
                return self.action

try:
    from rl.policy.Policy import Policy, spPolicy
except ImportError:
    try:
        from ..policy.Policy import Policy, spPolicy
    except (ImportError, ValueError):
        class Policy(Generic[S, A], ABC):  # type: ignore
            @abstractmethod
            def getAction(
                self,
                actionValues: Dict[A, float],
                actionSet: Set[A],
                maxAction: Optional[A] = None,
            ) -> Optional[A]:
                raise NotImplementedError

            def get_action(
                self,
                action_values: Dict[A, float],
                action_set: Set[A],
                max_action: Optional[A] = None,
            ) -> Optional[A]:
                return self.getAction(action_values, action_set, max_action)

        spPolicy = Policy  # type: ignore

try:
    from rl.policy.EpsilonGreedy import EpsilonGreedy, spEpsilonGreedy
    from rl.policy.EpsilonGreedyFactory import EpsilonGreedyFactory
except ImportError:
    try:
        from ..policy.EpsilonGreedy import EpsilonGreedy, spEpsilonGreedy
        from ..policy.EpsilonGreedyFactory import EpsilonGreedyFactory
    except (ImportError, ValueError):
        class EpsilonGreedy(Policy[S, A]):  # type: ignore
            def __init__(self, greediness: float) -> None:
                self.greediness = greediness

            def getAction(
                self,
                actionValues: Dict[A, float],
                actionSet: Set[A],
                maxAction: Optional[A] = None,
            ) -> Optional[A]:
                return next(iter(actionSet)) if actionSet else None

        spEpsilonGreedy = EpsilonGreedy  # type: ignore

        class EpsilonGreedyFactory(Generic[S, A]):  # type: ignore
            def __init__(self, greediness: float) -> None:
                self._instance: Policy[S, A] = EpsilonGreedy(greediness)

            def get(self) -> Policy[S, A]:
                return self._instance


def _get_default_learning_policy() -> Optional[Any]:
    try:
        return EpsilonGreedyFactory(DEFAULT_GREEDINESS_LEARNING_POLICY).get()
    except Exception:
        return None


class LearningAlgorithm(Generic[S, A], ABC):
    """
    Base class for all learning algorithms.

    All learning algorithm, reinforcement, supervised, unsupervised inherit
    from this class.
    """

    _default_learning_policy: Optional[Policy[Any, Any]] = _get_default_learning_policy()
    _defaultLearningPolicy: Optional[Policy[Any, Any]] = _default_learning_policy

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "get_action" in cls.__dict__ and "getAction" not in cls.__dict__:
            cls.getAction = cls.get_action  # type: ignore
        if "get_state_action_value" in cls.__dict__ and "getStateActionValue" not in cls.__dict__:
            cls.getStateActionValue = cls.get_state_action_value  # type: ignore

    def __init__(
        self,
        control_policy: Optional[Policy[S, A]] = None,
        *,
        controlPolicy: Optional[Policy[S, A]] = None,
    ) -> None:
        """
        :param control_policy: policy for action selection online.
        """
        policy = control_policy if control_policy is not None else controlPolicy
        self._control_policy: Policy[S, A] = policy  # type: ignore
        self._learning_policy: Optional[Policy[S, A]] = (
            self._default_learning_policy
            if self._default_learning_policy is not None
            else self._defaultLearningPolicy
        )
        self._default_state_action_value: float = float()

    @property
    def _controlPolicy(self) -> Policy[S, A]:
        return self._control_policy

    @_controlPolicy.setter
    def _controlPolicy(self, policy: Policy[S, A]) -> None:
        self._control_policy = policy

    @property
    def _learningPolicy(self) -> Optional[Policy[S, A]]:
        return self._learning_policy

    @_learningPolicy.setter
    def _learningPolicy(self, policy: Optional[Policy[S, A]]) -> None:
        self._learning_policy = policy

    @property
    def _defaultStateActionValue(self) -> float:
        return self._default_state_action_value

    @_defaultStateActionValue.setter
    def _defaultStateActionValue(self, value: float) -> None:
        self._default_state_action_value = value

    @abstractmethod
    def update(
        self,
        currentStateAction: StateAction[S, A],
        nextState: Optional[S],
        reward: float,
        actionSet: Set[A],
    ) -> None:
        """
        Update the stateAction map.

        :param currentStateAction: Current state-action pair.
        :param nextState: Next state.
        :param reward: Reward for the transition from current-state action to next state-action.
        :param actionSet: Set of possible actions.
        """
        raise NotImplementedError

    @abstractmethod
    def getAction(
        self,
        state: Optional[S],
        actionSet: Set[A],
    ) -> Optional[A]:
        """
        :param state: the state to take the action to.
        :param actionSet: set of possible actions.
        :return: action based on control policy and current state.
        """
        raise NotImplementedError

    def get_action(
        self,
        state: Optional[S],
        action_set: Set[A],
    ) -> Optional[A]:
        return self.getAction(state, action_set)

    @abstractmethod
    def getStateActionValue(
        self,
        stateAction: StateAction[S, A],
    ) -> float:
        """
        :param stateAction: State-action pair.
        :return: Value of stateAction.
        """
        raise NotImplementedError

    def get_state_action_value(
        self,
        state_action: StateAction[S, A],
    ) -> float:
        return self.getStateActionValue(state_action)

    def setControlPolicy(self, policy: Policy[S, A]) -> None:
        """
        :param policy: policy to set the current control policy with.
        """
        self._control_policy = policy

    def set_control_policy(self, policy: Policy[S, A]) -> None:
        self.setControlPolicy(policy)

    def getControlPolicy(self) -> Policy[S, A]:
        """
        :return: current control policy.
        """
        return self._control_policy

    def get_control_policy(self) -> Policy[S, A]:
        return self.getControlPolicy()

    def reset(self) -> None:
        """
        Reset anything prior to running episode.
        """
        pass

    def getDefaultStateActionValue(self) -> float:
        """
        :return: current default action value.
        """
        return self._default_state_action_value

    def get_default_state_action_value(self) -> float:
        return self.getDefaultStateActionValue()

    def setDefaultStateActionValue(self, defaultStateActionValue: float) -> None:
        """
        :param defaultStateActionValue: set the default state action value.
        """
        self._default_state_action_value = defaultStateActionValue

    def set_default_state_action_value(self, default_state_action_value: float) -> None:
        self.setDefaultStateActionValue(default_state_action_value)

    def setLearningPolicy(self, policy: Policy[S, A]) -> None:
        """
        :param policy: set the new learning policy.
        """
        self._learning_policy = policy

    def set_learning_policy(self, policy: Policy[S, A]) -> None:
        self.setLearningPolicy(policy)

    def getLearningPolicy(self) -> Optional[Policy[S, A]]:
        """
        :return: current learning policy.
        """
        return self._learning_policy

    def get_learning_policy(self) -> Optional[Policy[S, A]]:
        return self.getLearningPolicy()

    def serialize(self, file: str) -> None:
        """
        :param file: path to file to serialize to.
        """
        pass

    def deSerialize(self, file: str) -> None:
        """
        :param file: path to file to deserialize from.
        """
        pass

    def de_serialize(self, file: str) -> None:
        self.deSerialize(file)

    def _getLearningPolicyAction(
        self,
        actionValueMap: Dict[A, float],
        actionSet: Set[A],
        maxAction: Optional[A] = None,
    ) -> Optional[A]:
        """
        Returns the action selected by learning policy given action values from some state.

        :param actionValueMap: a mapping of action and their reward given current state.
        :param actionSet: current action set.
        :param maxAction: action with maximum value (optional optimization).
        :return: action selected by learning policy.
        """
        if self._learning_policy is None:
            return None
        action_fn = getattr(self._learning_policy, "getAction", None) or getattr(
            self._learning_policy, "get_action", None
        )
        if action_fn is not None:
            if maxAction is not None:
                return action_fn(actionValueMap, actionSet, maxAction)
            return action_fn(actionValueMap, actionSet)
        return None

    def _get_learning_policy_action(
        self,
        action_value_map: Dict[A, float],
        action_set: Set[A],
        max_action: Optional[A] = None,
    ) -> Optional[A]:
        return self._getLearningPolicyAction(action_value_map, action_set, max_action)


spLearningAlgorithm = LearningAlgorithm