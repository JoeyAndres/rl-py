"""rl - Reinforcement Learning
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

from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

S = TypeVar("S")
A = TypeVar("A")

StateAndReward = Tuple[S, float]
spStateAndReward = StateAndReward

try:
    from .actuator import Actuator
except ImportError:
    try:
        from .Actuator import Actuator
    except ImportError:
        try:
            from actuator import Actuator  # type: ignore
        except ImportError:
            try:
                from Actuator import Actuator  # type: ignore
            except ImportError:
                class Actuator(Generic[A]):  # type: ignore
                    pass

try:
    from .sensor import Sensor
except ImportError:
    try:
        from .Sensor import Sensor
    except ImportError:
        try:
            from sensor import Sensor  # type: ignore
        except ImportError:
            try:
                from Sensor import Sensor  # type: ignore
            except ImportError:
                class Sensor(Generic[S]):  # type: ignore
                    pass

try:
    from .state_action import StateAction
except ImportError:
    try:
        from .StateAction import StateAction
    except ImportError:
        try:
            from state_action import StateAction  # type: ignore
        except ImportError:
            try:
                from StateAction import StateAction  # type: ignore
            except ImportError:
                class StateAction(Generic[S, A]):  # type: ignore
                    def __init__(self, state: S, action: A) -> None:
                        self._state = state
                        self._action = action

                    @property
                    def state(self) -> S:
                        return self._state

                    @property
                    def action(self) -> A:
                        return self._action

                    def get_state(self) -> S:
                        return self._state

                    def getState(self) -> S:
                        return self._state

                    def get_action(self) -> A:
                        return self._action

                    def getAction(self) -> A:
                        return self._action


class Environment(ABC, Generic[S, A]):
    """A module that maps a state-action to next-state and reward.

    All Environment classes will override this class itself.
    """

    SA = StateAction

    def __init__(self, actuator: Actuator[A], sensor: Sensor[S]) -> None:
        self._actuator: Actuator[A] = actuator
        self._sensor: Sensor[S] = sensor

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "get_next_state_and_reward" in cls.__dict__ and "getNextStateAndReward" not in cls.__dict__:
            cls.getNextStateAndReward = cls.__dict__["get_next_state_and_reward"]
        elif "getNextStateAndReward" in cls.__dict__ and "get_next_state_and_reward" not in cls.__dict__:
            cls.get_next_state_and_reward = cls.__dict__["getNextStateAndReward"]

        if "apply_action" in cls.__dict__ and "applyAction" not in cls.__dict__:
            cls.applyAction = cls.__dict__["apply_action"]
        elif "applyAction" in cls.__dict__ and "apply_action" not in cls.__dict__:
            cls.apply_action = cls.__dict__["applyAction"]

        if "get_actuator" in cls.__dict__ and "getActuator" not in cls.__dict__:
            cls.getActuator = cls.__dict__["get_actuator"]
        elif "getActuator" in cls.__dict__ and "get_actuator" not in cls.__dict__:
            cls.get_actuator = cls.__dict__["getActuator"]

        if "get_sensor" in cls.__dict__ and "getSensor" not in cls.__dict__:
            cls.getSensor = cls.__dict__["get_sensor"]
        elif "getSensor" in cls.__dict__ and "get_sensor" not in cls.__dict__:
            cls.get_sensor = cls.__dict__["getSensor"]

        abstracts = set(getattr(cls, "__abstractmethods__", set()))
        if hasattr(cls, "getNextStateAndReward") and not getattr(cls.getNextStateAndReward, "__isabstractmethod__", False):
            abstracts.discard("getNextStateAndReward")
            abstracts.discard("get_next_state_and_reward")
        cls.__abstractmethods__ = frozenset(abstracts)

    @abstractmethod
    def get_next_state_and_reward(
        self, state_action: StateAction[S, A]
    ) -> Tuple[S, float]:
        """Given this state-action, gives next state and reward.

        :param state_action: Given this state-action, gives next state and reward.
        :return: Next state and reward.
        """
        pass

    @abstractmethod
    def getNextStateAndReward(
        self, stateAction: StateAction[S, A]
    ) -> Tuple[S, float]:
        """Given this state-action, gives next state and reward.

        :param stateAction: Given this state-action, gives next state and reward.
        :return: Next state and reward.
        """
        pass

    def apply_action(self, action: A) -> Tuple[S, float]:
        """Apply action to environment.

        :param action: Action to be applied to environment.
        :return: Next state and reward.
        """
        if hasattr(self._sensor, "get_last_observed_state"):
            current_state = self._sensor.get_last_observed_state()
        elif hasattr(self._sensor, "getLastObservedState"):
            current_state = self._sensor.getLastObservedState()
        else:
            current_state = getattr(self._sensor, "last_observed_state", None)

        current_state_action = StateAction(current_state, action)
        next_state_and_reward = self.get_next_state_and_reward(current_state_action)

        if hasattr(self._sensor, "set_last_observed_state_and_reward"):
            self._sensor.set_last_observed_state_and_reward(next_state_and_reward)
        elif hasattr(self._sensor, "setLastObservedStateAndReward"):
            self._sensor.setLastObservedStateAndReward(next_state_and_reward)

        return next_state_and_reward

    def applyAction(self, action: A) -> Tuple[S, float]:
        """Apply action to environment.

        :param action: Action to be applied to environment.
        :return: Next state and reward.
        """
        return self.apply_action(action)

    @property
    def actuator(self) -> Actuator[A]:
        return self._actuator

    @actuator.setter
    def actuator(self, actuator: Actuator[A]) -> None:
        self._actuator = actuator

    def get_actuator(self) -> Actuator[A]:
        return self._actuator

    def getActuator(self) -> Actuator[A]:
        return self.get_actuator()

    @property
    def sensor(self) -> Sensor[S]:
        return self._sensor

    @sensor.setter
    def sensor(self, sensor: Sensor[S]) -> None:
        self._sensor = sensor

    def get_sensor(self) -> Sensor[S]:
        return self._sensor

    def getSensor(self) -> Sensor[S]:
        return self.get_sensor()

    def reset(self) -> None:
        """Call when environment is reset."""
        self._sensor.reset()


spEnvironment = Environment