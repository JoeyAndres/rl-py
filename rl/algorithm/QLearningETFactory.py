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

try:
    from .QLearningET import QLearningET
    from .ReinforcementLearningFactory import ReinforcementLearningFactory
    from ..policy.Policy import Policy
except ImportError:
    from rl.algorithm.QLearningET import QLearningET
    from rl.algorithm.ReinforcementLearningFactory import ReinforcementLearningFactory
    from rl.policy.Policy import Policy

S = TypeVar("S")
A = TypeVar("A")


class QLearningETFactory(ReinforcementLearningFactory[S, A], Generic[S, A]):
    """Factory method for QLearningET.

    Args:
        step_size (float): Learning rate / step size.
        discount_rate (float): Discount rate factor.
        policy (Policy[S, A]): Action selection policy.
        lambda_ (float): Eligibility trace decay rate parameter.
    """

    def __init__(
        self,
        step_size: float,
        discount_rate: float,
        policy: Policy[S, A],
        lambda_: float,
    ) -> None:
        super().__init__()
        self._instance: QLearningET[S, A] = QLearningET(
            step_size, discount_rate, policy, lambda_
        )