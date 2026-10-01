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

from typing import ClassVar, Optional, Type

from ...coding.TileCode import TileCode
from ...declares import FLOAT
from ...policy.Policy import Policy
from ..LearningAlgorithmFactory import LearningAlgorithmFactory
from .ReinforcementLearningGDAbstract import ReinforcementLearningGDAbstract


class ReinforcementLearningGDFactory(LearningAlgorithmFactory):
    """Factory method for ReinforcementLearningGD. Subclasses set ALGORITHM_CLASS."""

    ALGORITHM_CLASS: ClassVar[Optional[Type[ReinforcementLearningGDAbstract]]] = None

    def __init__(
        self,
        tileCode: TileCode,
        stepSize: FLOAT,
        discountRate: FLOAT,
        lambda_: FLOAT,
        policy: Policy,
    ) -> None:
        super().__init__()
        if self.ALGORITHM_CLASS is None:
            raise TypeError(f"{type(self).__name__} must define ALGORITHM_CLASS.")
        self._instance = self.ALGORITHM_CLASS(
            tileCode, stepSize, discountRate, lambda_, policy
        )
