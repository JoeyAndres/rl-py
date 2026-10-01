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

from rl.agent import Sensor

POS = 0
VEL = 1


class SensorMountainCar(Sensor):
    def __init__(self):
        super().__init__([0.0, 0.0])

    def isTerminalState(self, stateData) -> bool:
        pos = stateData.at(POS) if hasattr(stateData, "at") else stateData[POS]
        if abs(pos - 0.50) <= 0.01:
            return True
        return False


class SensorMountainCarFactory:
    @staticmethod
    def create() -> SensorMountainCar:
        return SensorMountainCar()