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

from typing import Any, Generic, Optional, Tuple, TypeVar

S = TypeVar("S")
A = TypeVar("A")

MAX_EPISODES: int = 100000
FLOAT = float
UINT = int

try:
    from rl.agent.ActionContainer import ActionContainer
    from rl.agent.Environment import Environment
    from rl.agent.StateAction import StateAction
    from rl.algorithm.LearningAlgorithm import LearningAlgorithm
except (ImportError, ModuleNotFoundError):
    try:
        from rl.agent.action_container import ActionContainer
        from rl.agent.environment import Environment
        from rl.agent.state_action import StateAction
        from rl.algorithm.learning_algorithm import LearningAlgorithm
    except (ImportError, ModuleNotFoundError):
        try:
            from .ActionContainer import ActionContainer
            from .Environment import Environment
            from .StateAction import StateAction
            from ..algorithm.LearningAlgorithm import LearningAlgorithm
        except (ImportError, ModuleNotFoundError, ValueError):
            try:
                from .action_container import ActionContainer
                from .environment import Environment
                from .state_action import StateAction
                from ..algorithm.learning_algorithm import LearningAlgorithm
            except (ImportError, ModuleNotFoundError, ValueError):
                ActionContainer = Any  # type: ignore[assignment,misc]
                Environment = Any  # type: ignore[assignment,misc]
                LearningAlgorithm = Any  # type: ignore[assignment,misc]

                class StateAction(Generic[S, A]):  # type: ignore[no-redef]
                    def __init__(self, state: S, action: A) -> None:
                        self.state: S = state
                        self.action: A = action

                    def getState(self) -> S:
                        return self.state

                    def getAction(self) -> A:
                        return self.action

                    def get_state(self) -> S:
                        return self.state

                    def get_action(self) -> A:
                        return self.action


def _get_action_set(container: Any) -> Any:
    if hasattr(container, "get_action_set"):
        return container.get_action_set()
    return container.getActionSet()


def _get_action(algorithm: Any, state: Any, action_set: Any) -> Any:
    if hasattr(algorithm, "get_action"):
        return algorithm.get_action(state, action_set)
    return algorithm.getAction(state, action_set)


class AgentSupervised(Generic[S, A]):
    """A class that represents an rl agent for supervised learning.

    Supervised learning agents don't interact with the environment,
    but only take data from the environment.
    """

    def __init__(
        self,
        action_container: Any,
        learning_algorithm: Any,
    ) -> None:
        """
        :param action_container: aggregates an ActionContainer object.
        :param learning_algorithm: aggregates a LearningAlgorithm object.
        """
        self._action_container: Any = action_container
        self._learning_algorithm: Any = learning_algorithm

    @property
    def _actionContainer(self) -> Any:
        return self._action_container

    @_actionContainer.setter
    def _actionContainer(self, value: Any) -> None:
        self._action_container = value

    @property
    def _learningAlgorithm(self) -> Any:
        return self._learning_algorithm

    @_learningAlgorithm.setter
    def _learningAlgorithm(self, value: Any) -> None:
        self._learning_algorithm = value

    def train(
        self,
        state: S,
        action: A,
        reward: float,
        next_state: S,
    ) -> None:
        """For training (instead of control) purposes.

        :param state: Current state.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Resulting next state.
        """
        action_set = _get_action_set(self._action_container)
        self._learning_algorithm.update(
            StateAction(state, action),
            next_state,
            reward,
            action_set,
        )


class AgentSupervisedGD(AgentSupervised[Tuple[float, ...], Tuple[float, ...]]):
    """AgentSupervised for Gradient Descent."""
    pass


class Agent(Generic[S, A]):
    """A class that represents an rl agent.

    In Artificial Intelligence, Agent is the entity that receives input from
    environment. In response, it outputs action to environment.
    """

    def __init__(
        self,
        environment: Any,
        learning_algorithm: Any,
    ) -> None:
        """
        :param environment: aggregates an Environment object.
        :param learning_algorithm: aggregates a LearningAlgorithm object.
        """
        self._environment: Any = environment
        self._learning_algorithm: Any = learning_algorithm
        self._accumulative_reward: float = 0.0

        self._current_state: Optional[S] = self.get_last_observed_state()
        action_set = _get_action_set(self._get_actuator())
        self._current_action: Optional[A] = _get_action(
            self._learning_algorithm, self._current_state, action_set
        )

    def _get_sensor(self) -> Any:
        if hasattr(self._environment, "get_sensor"):
            return self._environment.get_sensor()
        return self._environment.getSensor()

    def _get_actuator(self) -> Any:
        if hasattr(self._environment, "get_actuator"):
            return self._environment.get_actuator()
        return self._environment.getActuator()

    @property
    def _learningAlgorithm(self) -> Any:
        return self._learning_algorithm

    @_learningAlgorithm.setter
    def _learningAlgorithm(self, value: Any) -> None:
        self._learning_algorithm = value

    @property
    def _currentState(self) -> Optional[S]:
        return self._current_state

    @_currentState.setter
    def _currentState(self, value: Optional[S]) -> None:
        self._current_state = value

    @property
    def _currentAction(self) -> Optional[A]:
        return self._current_action

    @_currentAction.setter
    def _currentAction(self, value: Optional[A]) -> None:
        self._current_action = value

    @property
    def _accumulativeReward(self) -> float:
        return self._accumulative_reward

    @_accumulativeReward.setter
    def _accumulativeReward(self, value: float) -> None:
        self._accumulative_reward = value

    def get_last_observed_state(self) -> S:
        """Return current state of the agent."""
        sensor = self._get_sensor()
        if hasattr(sensor, "get_last_observed_state"):
            return sensor.get_last_observed_state()
        return sensor.getLastObservedState()

    def getLastObservedState(self) -> S:
        return self.get_last_observed_state()

    def train(
        self,
        state: S,
        action: A,
        reward: float,
        next_state: S,
    ) -> None:
        """For training (instead of control) purposes.

        :param state: Current state.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Resulting next state.
        """
        action_set = _get_action_set(self._get_actuator())
        self._learning_algorithm.update(
            StateAction(state, action),
            next_state,
            reward,
            action_set,
        )

    def pre_execute(self) -> None:
        """Prepare agent prior to start execution."""
        self._learning_algorithm.reset()
        self._current_state = self.get_last_observed_state()
        action_set = _get_action_set(self._get_actuator())
        self._current_action = _get_action(
            self._learning_algorithm, self._current_state, action_set
        )
        self._accumulative_reward = 0.0

    def preExecute(self) -> None:
        self.pre_execute()

    def execute(self) -> None:
        """Execute a single time step."""
        if self._current_action is not None:
            self.apply_action(self._current_action)
        next_state = self.get_last_observed_state()
        sensor = self._get_sensor()
        if hasattr(sensor, "get_last_observed_reward"):
            reward = float(sensor.get_last_observed_reward())
        else:
            reward = float(sensor.getLastObservedReward())

        # Accumulate reward.
        self._accumulative_reward += reward

        # Update value, e.g. perform backups.
        if self._current_state is not None and self._current_action is not None:
            self.train(self._current_state, self._current_action, reward, next_state)

        # Get the action from the algorithm.
        action_set = _get_action_set(self._get_actuator())
        self._current_action = _get_action(
            self._learning_algorithm, next_state, action_set
        )

        # Update current state.
        self._current_state = next_state

    def execute_episode(self, max_iter: int = MAX_EPISODES) -> int:
        """Execute many timesteps until end of episode.

        :param max_iter: Maximum iteration.
        :return: Number of iterations to achieve the task.
        """
        self.pre_execute()
        i = 0
        while i < max_iter and not self.episode_done():
            self.execute()
            i += 1
        self.post_execute()
        return i

    def executeEpisode(self, maxIter: int = MAX_EPISODES) -> int:
        return self.execute_episode(max_iter=maxIter)

    def episode_done(self) -> bool:
        """Return True if episode is done."""
        sensor = self._get_sensor()
        if hasattr(sensor, "is_terminal_state"):
            return bool(sensor.is_terminal_state(self._current_state))
        return bool(sensor.isTerminalState(self._current_state))

    def episodeDone(self) -> bool:
        return self.episode_done()

    def post_execute(self) -> float:
        """Does cleanup routines after reaching terminal state.

        :return: Cumulative reward.
        """
        return self._accumulative_reward

    def postExecute(self) -> float:
        return self.post_execute()

    def get_accumulative_reward(self) -> float:
        """Acquire the accumulated reward so far within the episode."""
        return self._accumulative_reward

    def getAccumulativeReward(self) -> float:
        return self.get_accumulative_reward()

    def apply_action(self, action: A) -> None:
        """Agent applies action to the environment."""
        if hasattr(self._environment, "apply_action"):
            self._environment.apply_action(action)
        else:
            self._environment.applyAction(action)

    def applyAction(self, action: A) -> None:
        self.apply_action(action)

    def reset(self) -> None:
        """Calls the Environment reset."""
        self._environment.reset()

    def get_last_action_applied(self) -> Optional[A]:
        """Return last action applied."""
        return self._current_action

    def getLastActionApplied(self) -> Optional[A]:
        return self.get_last_action_applied()


class AgentGD(Agent[Tuple[float, ...], Tuple[float, ...]]):
    """Agent for Gradient Descent."""
    pass