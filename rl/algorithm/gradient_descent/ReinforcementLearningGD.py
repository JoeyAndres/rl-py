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

from ...coding.TileCode import TileCode
from ...declares import FLOAT
from ...policy.Policy import Policy
from .GradientDescentTileCode import GradientDescentTileCode
from .ReinforcementLearningGDAbstract import ReinforcementLearningGDAbstract


class ReinforcementLearningGD(ReinforcementLearningGDAbstract):
    """Reinforcement learning with tile-coded gradient descent."""

    def __init__(
        self,
        tileCode: TileCode,
        stepSize: FLOAT,
        discountRate: FLOAT,
        lambda_: FLOAT,
        policy: Policy,
    ) -> None:
        super().__init__(tileCode, stepSize, discountRate, lambda_, policy)
        self._gradientDescent = GradientDescentTileCode(tileCode, stepSize, discountRate, lambda_)
