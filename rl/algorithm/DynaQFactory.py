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

from .ReinforcementLearningFactory import ReinforcementLearningFactory
from .DynaQ import DynaQ
from ..policy.Policy import Policy

S = TypeVar("S")
A = TypeVar("A")


class DynaQFactory(ReinforcementLearningFactory[S, A], Generic[S, A]):
    """Factory method for DynaQ."""

    def __init__(
        self,
        step_size: float,
        discount_rate: float,
        policy: Policy[S, A],
        simulation_iteration_count: int,
        state_transition_greediness: float,
        state_transition_step_size: float,
    ) -> None:
        super().__init__()
        self._instance = DynaQ(
            step_size,
            discount_rate,
            policy,
            simulation_iteration_count,
            state_transition_greediness,
            state_transition_step_size,
        )