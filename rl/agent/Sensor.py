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

from abc import ABC, abstractmethod
import math
from typing import Generic, Tuple, TypeVar

S = TypeVar("S")

FLOAT = float
spState = S
spStateAndReward = Tuple[S, FLOAT]


class Sensor(ABC, Generic[S]):
    """Base and interface class for all Sensor objects.

    For sensors with states of discrete nature, see SensorDiscrete.
    Otherwise, use this or SensorContinuous.
    """

    def __init__(self, initialState: S) -> None:
        """Initialize Sensor with initial state.

        :param initialState: Initial state.
        """
        self._lastObservedReward: float = math.nan
        self._initialState: S = initialState
        self._lastObservedState: S = initialState

    def getLastObservedState(self) -> S:
        """Return current state of agent in environment.

        :return: Current state of agent in environment.
        """
        return self._lastObservedState

    get_last_observed_state = getLastObservedState

    def getLastObservedReward(self) -> float:
        """Maps sensorState to its corresponding reward.

        :return: reward
        """
        return self._lastObservedReward

    get_last_observed_reward = getLastObservedReward

    def setLastObservedState(self, s: S) -> None:
        """Set the last observed state.

        :param s: Last observed state.
        """
        self._lastObservedState = s

    set_last_observed_state = setLastObservedState

    def setInitialState(self, s: S) -> None:
        """Changes initial state.

        :param s: New initial state.
        """
        self._initialState = s

    set_initial_state = setInitialState

    def setLastObservedReward(self, r: float) -> None:
        """Set the last observed reward.

        :param r: Last observed reward.
        """
        self._lastObservedReward = r

    set_last_observed_reward = setLastObservedReward

    def setLastObservedStateAndReward(
        self, stateAndReward: Tuple[S, float]
    ) -> None:
        """Set last observed state and reward.

        :param stateAndReward: pair of state and reward.
        """
        self.setLastObservedState(stateAndReward[0])
        self.setLastObservedReward(stateAndReward[1])

    set_last_observed_state_and_reward = setLastObservedStateAndReward

    def reset(self) -> None:
        """Resets last observed state to initial state."""
        self._lastObservedState = self._initialState

    @abstractmethod
    def isTerminalState(self, stateData: S) -> bool:
        """Determine if it is a terminal state.

        :param stateData: State data to determine if it is a terminal state.
        :return: True if its a terminal state.
        """
        pass

    def is_terminal_state(self, state_data: S) -> bool:
        """Determine if it is a terminal state (snake_case alias)."""
        return self.isTerminalState(state_data)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "is_terminal_state" in cls.__dict__ and "isTerminalState" not in cls.__dict__:
            cls.isTerminalState = cls.__dict__["is_terminal_state"]
        elif "isTerminalState" in cls.__dict__ and "is_terminal_state" not in cls.__dict__:
            cls.is_terminal_state = cls.__dict__["isTerminalState"]


spSensor = Sensor