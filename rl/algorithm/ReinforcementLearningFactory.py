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

import sys
from typing import Generic, TypeVar

from ..agent.StateActionPairContainer import StateActionPairContainerSimple
from ..utility.FactoryAbstract import FactoryAbstract
from .LearningAlgorithmFactory import LearningAlgorithmFactory
from .ReinforcementLearning import ReinforcementLearning

S = TypeVar("S")
A = TypeVar("A")



class ReinforcementLearningFactory(
    FactoryAbstract[ReinforcementLearning[S, A]],
    Generic[S, A],
):
    """
    Factory method for ReinforcementLearning. Can't really use this.

    :tparam S: State data type.
    :tparam A: Action data type.
    """

    def __init__(self) -> None:
        super().__init__()