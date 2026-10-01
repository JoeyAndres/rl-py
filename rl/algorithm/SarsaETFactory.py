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

from typing import Generic, TypeVar

from rl.algorithm.reinforcement_learning_factory import ReinforcementLearningFactory
from rl.algorithm.sarsa_et import SarsaET
from rl.policy.policy import Policy

S = TypeVar("S")
A = TypeVar("A")


class SarsaETFactory(ReinforcementLearningFactory[S, A], Generic[S, A]):
    """Factory method for SarsaET.

    :param S: State data type.
    :param A: Action data type.
    """

    def __init__(
        self,
        step_size: float,
        discount_rate: float,
        policy: Policy[S, A],
        lambda_: float,
    ) -> None:
        super().__init__()
        self._instance: SarsaET[S, A] = SarsaET(
            step_size, discount_rate, policy, lambda_
        )