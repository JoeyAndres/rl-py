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

from typing import Any, Generic, List, Optional, Type, TypeVar

from .ReinforcementLearningGDFactory import ReinforcementLearningGDFactory
from .QLearningGD import QLearningGD

from ...coding.TileCode import DEFAULT_TILE_CONT

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")
STATE_DIM = TypeVar("STATE_DIM")


class QLearningGDFactory(
    ReinforcementLearningGDFactory[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM, QLearningGD],
    Generic[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM],
):
    """Factory method for QLearningGD.

    :tparam D: Number of dimension.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
    :tparam STATE_DIM: Number of dimension in state.
                      Implies that action is D - STATE_DIM.
    """

    algorithm_class: Type[Any] = QLearningGD

    def __init__(
        self,
        tile_code: Any,
        step_size: float,
        discount_rate: float,
        lambda_: float,
        policy: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize QLearningGDFactory.

        :param tile_code: The tile coding instance.
        :param step_size: Step size (alpha).
        :param discount_rate: Discount rate (gamma).
        :param lambda_: Eligibility trace decay parameter (lambda).
        :param policy: Action selection policy.
        """
        try:
            super().__init__(
                tile_code,
                step_size,
                discount_rate,
                lambda_,
                policy,
                *args,
                algorithm_class=QLearningGD,
                **kwargs,
            )
        except TypeError:
            super().__init__(
                tile_code,
                step_size,
                discount_rate,
                lambda_,
                policy,
                *args,
                **kwargs,
            )
        if getattr(self, "_instance", None) is None:
            self._instance = QLearningGD(
                tile_code,
                step_size,
                discount_rate,
                lambda_,
                policy,
            )

    def get(self) -> Any:
        """Get the created QLearningGD instance."""
        if hasattr(super(), "get"):
            try:
                return super().get()
            except (AttributeError, NotImplementedError):
                pass
        return getattr(self, "_instance", None)


__all__ = ["QLearningGDFactory"]