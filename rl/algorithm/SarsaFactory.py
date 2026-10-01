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

from typing import Generic, Optional, TypeVar

from rl.agent.StateActionPairContainer import StateActionPairContainerSimple
from rl.algorithm.ReinforcementLearning import ReinforcementLearning
from rl.algorithm.ReinforcementLearningFactory import ReinforcementLearningFactory
from rl.algorithm.Sarsa import Sarsa
from rl.policy.Policy import Policy

S = TypeVar("S")
A = TypeVar("A")
SAP_CONTAINER = TypeVar("SAP_CONTAINER")


class SarsaFactory(ReinforcementLearningFactory[S, A, SAP_CONTAINER], Generic[S, A, SAP_CONTAINER]):
    """Factory method for Sarsa.

    :tparam S: State data type.
    :tparam A: Action data type.
    """

    def __init__(
        self,
        step_size: float,
        discount_rate: float,
        policy: Policy[S, A],
    ) -> None:
        super().__init__()
        self._instance: Optional[ReinforcementLearning[S, A, SAP_CONTAINER]] = Sarsa(
            step_size, discount_rate, policy
        )