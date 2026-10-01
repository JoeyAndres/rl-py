# rl - Reinforcement Learning
# Copyright (C) 2016  Joey Andres<yeojserdna@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

FLOAT = float

S = TypeVar("S")
A = TypeVar("A")
StateParam = TypeVar("StateParam")
ActionParam = TypeVar("ActionParam")


def concat_array(a1: Sequence[Any], a2: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(a1) + tuple(a2)


concatArray = concat_array


try:
    from rl.agent.StateAction import StateAction
except ImportError:
    class StateAction(Generic[S, A]):
        def __init__(self, state: S, action: A) -> None:
            self._state: S = state
            self._action: A = action

        def get_state(self) -> S:
            return self._state

        def getState(self) -> S:
            return self._state

        def get_action(self) -> A:
            return self._action

        def getAction(self) -> A:
            return self._action

        def set_state(self, state: S) -> None:
            self._state = state

        def setState(self, state: S) -> None:
            self._state = state

        def set_action(self, action: A) -> None:
            self._action = action

        def setAction(self, action: A) -> None:
            self._action = action


try:
    from rl.policy.Policy import Policy
except ImportError:
    class Policy(Generic[S, A], ABC):
        @abstractmethod
        def get_action(
            self,
            action_value_map: Dict[A, float],
            action_set: Set[A],
            max_action: Optional[A] = None,
        ) -> A:
            pass

        def getAction(
            self,
            action_value_map: Dict[A, float],
            action_set: Set[A],
            max_action: Optional[A] = None,
        ) -> A:
            return self.get_action(action_value_map, action_set, max_action)

        def reset(self) -> None:
            pass


try:
    from rl.coding.CourseCode import CourseCode
except ImportError:
    class CourseCode(Generic[S]):
        pass


try:
    from rl.algorithm.gradient_descent.GradientDescentAbstract import (
        GradientDescentAbstract,
    )
except ImportError:
    class GradientDescentAbstract(Generic[StateParam, ActionParam], ABC):
        ACTION_DIM: int = 0

        @abstractmethod
        def build_action_values(
            self, action_set: Set[ActionParam], next_state: StateParam
        ) -> Tuple[Dict[ActionParam, float], ActionParam]:
            pass

        buildActionValues = build_action_values

        @abstractmethod
        def update_weights(
            self,
            current_state: StateParam,
            current_action: ActionParam,
            next_state: StateParam,
            next_action_value: float,
            reward: float,
        ) -> None:
            pass

        updateWeights = update_weights

        @abstractmethod
        def get_value_from_parameters(self, params: Sequence[float]) -> float:
            pass

        getValueFromParameters = get_value_from_parameters


try:
    from rl.algorithm.LearningAlgorithm import LearningAlgorithm
except ImportError:
    class LearningAlgorithm(Generic[S, A], ABC):
        def __init__(self, control_policy: Optional[Policy[S, A]] = None) -> None:
            self._control_policy: Optional[Policy[S, A]] = control_policy
            self._learning_policy: Optional[Policy[S, A]] = control_policy
            self._default_state_action_value: float = 0.0

        @abstractmethod
        def update(
            self,
            current_state_action: StateAction[S, A],
            next_state: S,
            reward: float,
            action_set: Set[A],
        ) -> None:
            pass

        @abstractmethod
        def get_action(self, state: S, action_set: Set[A]) -> A:
            pass

        getAction = get_action

        @abstractmethod
        def get_state_action_value(self, state_action: StateAction[S, A]) -> float:
            pass

        getStateActionValue = get_state_action_value

        def set_control_policy(self, policy: Policy[S, A]) -> None:
            self._control_policy = policy

        setControlPolicy = set_control_policy

        def get_control_policy(self) -> Optional[Policy[S, A]]:
            return self._control_policy

        getControlPolicy = get_control_policy

        def set_learning_policy(self, policy: Policy[S, A]) -> None:
            self._learning_policy = policy

        setLearningPolicy = set_learning_policy

        def get_learning_policy(self) -> Optional[Policy[S, A]]:
            return self._learning_policy

        getLearningPolicy = get_learning_policy

        def reset(self) -> None:
            if self._control_policy is not None:
                self._control_policy.reset()
            if self._learning_policy is not None:
                self._learning_policy.reset()

        def _get_learning_policy_action(
            self,
            action_value_map: Dict[A, float],
            action_set: Set[A],
            max_action: Optional[A] = None,
        ) -> A:
            if self._learning_policy is not None:
                return self._learning_policy.get_action(
                    action_value_map, action_set, max_action
                )
            if max_action is not None:
                return max_action
            return next(iter(action_set))

        _getLearningPolicyAction = _get_learning_policy_action


class ReinforcementLearningGDAbstract(
    LearningAlgorithm[StateParam, ActionParam],
    Generic[StateParam, ActionParam],
):
    """
    Gradient descent implementation of Reinforcement Learning.
    """

    ACTION_DIM: int = 0

    StateParam = StateParam
    ActionParam = ActionParam

    def __init__(
        self,
        tile_code: Optional[Any],
        step_size: float,
        discount_rate: float,
        lambda_: float,
        policy: Optional[Policy[StateParam, ActionParam]] = None,
        d: Optional[int] = None,
        state_dim: Optional[int] = None,
    ) -> None:
        """
        :param tile_code: tileCode implementation to be aggregated.
        :param step_size: step size indicates how fast increment to a new estimates occur.
                          Too big will be fast but will likely cause overestimate. Too low
                          will be slow but will likely cause more precise estimates.
        :param discount_rate: How much current state influence future states.
        :param lambda_: How much current state influence future states (multiplied with discount rate).
        :param policy: Control policy.
        """
        super().__init__(policy)
        self.tile_code = tile_code
        self.step_size = step_size
        self.discount_rate = discount_rate
        self.lambda_ = lambda_
        self.d = d
        self.state_dim = state_dim
        if d is not None and state_dim is not None:
            self.action_dim = d - state_dim
        else:
            self.action_dim = self.ACTION_DIM

        self._gradient_descent: Optional[
            GradientDescentAbstract[StateParam, ActionParam]
        ] = None

    def _build_action_values(
        self,
        action_set: Set[ActionParam],
        next_state: StateParam,
    ) -> Tuple[Dict[ActionParam, float], ActionParam]:
        """
        Acquires the action-value map as well as the action with max action.

        :param action_set: Set of actions.
        :param next_state: Next state.
        :return: (action_value_map, max_action)
        """
        if self._gradient_descent is None:
            raise RuntimeError("Gradient descent object is not initialized.")
        if hasattr(self._gradient_descent, "build_action_values"):
            return self._gradient_descent.build_action_values(action_set, next_state)
        return self._gradient_descent.buildActionValues(action_set, next_state)

    _buildActionValues = _build_action_values

    def update(
        self,
        current_state_action: StateAction[StateParam, ActionParam],
        next_state_vector: StateParam,
        reward: float,
        action_set: Set[ActionParam],
    ) -> None:
        """
        :param current_state_action: current state-action vector to apply.
        :param next_state_vector: resulting state by taking state-action pair above.
        :param reward: reward for going to nextStateVector from currentState.
        :param action_set: set of possible actions.
        """
        action_value_map, max_action = self._build_action_values(
            action_set, next_state_vector
        )

        next_action = self._get_learning_policy_action(
            action_value_map, action_set, max_action
        )

        if self._gradient_descent is None:
            raise RuntimeError("Gradient descent object is not initialized.")

        current_state = (
            current_state_action.get_state()
            if hasattr(current_state_action, "get_state")
            else getattr(current_state_action, "state", current_state_action)
        )
        current_action = (
            current_state_action.get_action()
            if hasattr(current_state_action, "get_action")
            else getattr(current_state_action, "action", None)
        )

        if hasattr(self._gradient_descent, "update_weights"):
            self._gradient_descent.update_weights(
                current_state,
                current_action,
                next_state_vector,
                action_value_map[next_action],
                reward,
            )
        else:
            self._gradient_descent.updateWeights(
                current_state,
                current_action,
                next_state_vector,
                action_value_map[next_action],
                reward,
            )

    def get_action(
        self,
        state: StateParam,
        action_set: Set[ActionParam],
    ) -> ActionParam:
        """
        :param state: State to take action to.
        :param action_set: Set of possible actions.
        :return: Action determined by Control Policy.
        """
        action_value_map, max_action = self._build_action_values(action_set, state)
        if self._learning_policy is not None:
            if hasattr(self._learning_policy, "get_action"):
                return self._learning_policy.get_action(
                    action_value_map, action_set, max_action
                )
            return self._learning_policy.getAction(
                action_value_map, action_set, max_action
            )
        return max_action

    getAction = get_action

    def get_state_action_value(
        self,
        state_action: StateAction[StateParam, ActionParam],
    ) -> float:
        """
        :param state_action: State-action pair to determine value of.
        :return: value of state-action pair.
        """
        return self._get_state_action_value(state_action)

    getStateActionValue = get_state_action_value

    def _get_state_action_value(
        self,
        state_action: StateAction[StateParam, ActionParam],
    ) -> float:
        """
        :param state_action: State-action pair to determine value of.
        :return: value of state-action pair.
        """
        state_params = (
            state_action.get_state()
            if hasattr(state_action, "get_state")
            else getattr(state_action, "state", state_action)
        )
        action_params = (
            state_action.get_action()
            if hasattr(state_action, "get_action")
            else getattr(state_action, "action", None)
        )

        current_params = concat_array(state_params, action_params)

        if self._gradient_descent is None:
            raise RuntimeError("Gradient descent object is not initialized.")

        if hasattr(self._gradient_descent, "get_value_from_parameters"):
            return self._gradient_descent.get_value_from_parameters(current_params)
        return self._gradient_descent.getValueFromParameters(current_params)

    _getStateActionValue = _get_state_action_value

    def reset(self) -> None:
        """
        Reset routine for an algorithm for every episode.
        """
        super().reset()


spReinforcementLearningGDAbstract = Optional[ReinforcementLearningGDAbstract]