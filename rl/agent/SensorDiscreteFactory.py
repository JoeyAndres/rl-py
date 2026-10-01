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

from typing import Generic, TypeVar
from ..utility.FactoryAbstract import FactoryAbstract
from .SensorDiscrete import SensorDiscrete
from .Sensor import Sensor

S = TypeVar("S")


class SensorDiscreteFactory(FactoryAbstract[SensorDiscrete[S]], Generic[S]):
    """Factory object for SensorDiscrete.

    See Also:
        SensorDiscrete

    Args:
        S: SensorDiscrete data type.
    """

    def __init__(self, initialState: S) -> None:
        """
        See Also:
            SensorDiscrete

        Args:
            initialState: Initial state of the sensor.
        """
        super().__init__()
        self._instance: SensorDiscrete[S] = SensorDiscrete(initialState)