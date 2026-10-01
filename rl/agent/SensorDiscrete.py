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
from typing import Generic, Optional, Set, Tuple, TypeVar

S = TypeVar("S")

from .Sensor import Sensor


class SensorDiscrete(Sensor[S]):
    """
    Base class for sensors of discrete nature.
    """

    def __init__(self, initial_state: S) -> None:
        super().__init__(initial_state)
        self._terminal_states: Set[S] = set()

    def is_terminal_state(self, state_data: S) -> bool:
        """
        Check if state_data is a terminal state.

        :param state_data: State data to check.
        :return: True if it is a terminal state, False otherwise.
        """
        return state_data in self._terminal_states

    def isTerminalState(self, stateData: S) -> bool:
        """
        Compatibility alias for is_terminal_state.
        """
        return self.is_terminal_state(stateData)

    def add_terminal_state(self, terminal_data: S) -> None:
        """
        Add a new terminal state.

        :param terminal_data: New terminal state to be added.
        """
        self._terminal_states.add(terminal_data)

    def addTerminalState(self, terminalData: S) -> None:
        """
        Compatibility alias for add_terminal_state.
        """
        self.add_terminal_state(terminalData)


spSensorDiscrete = SensorDiscrete

__all__ = ["SensorDiscrete", "spSensorDiscrete"]