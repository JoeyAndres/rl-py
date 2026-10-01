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

from typing import (
    Any,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
)

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")
STATE_DIM = TypeVar("STATE_DIM")

FLOAT = float
FEATURE_VECTOR = Sequence[int]


def concat_array(a: Sequence[Any], b: Sequence[Any]) -> List[Any]:
    return list(a) + list(b)


concatArray = concat_array

try:
    from .GradientDescentTileCodeAbstract import GradientDescentTileCodeAbstract
except ImportError:
    try:
        from rl.algorithm.gradient_descent.GradientDescentTileCodeAbstract import (
            GradientDescentTileCodeAbstract,
        )
    except ImportError:
        try:
            from GradientDescentTileCodeAbstract import (
                GradientDescentTileCodeAbstract,
            )
        except ImportError:
            class GradientDescentTileCodeAbstract(
                Generic[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM]
            ):
                def __init__(
                    self,
                    tile_code: Any,
                    step_size: float,
                    discount_rate: float,
                    lambda_: float,
                ) -> None:
                    self._tile_code = tile_code
                    self._tileCode = tile_code
                    self._step_size = step_size
                    self._stepSize = step_size
                    self._discount_rate = discount_rate
                    self._discountRate = discount_rate
                    self._lambda = lambda_

                def get_value_from_feature_vector(
                    self, fv: Sequence[int]
                ) -> float:
                    return float(self._tile_code.get_value_from_feature_vector(fv))

                def getValueFromFeatureVector(
                    self, fv: Sequence[int]
                ) -> float:
                    return self.get_value_from_feature_vector(fv)

                def get_feature_vector(
                    self, parameters: Sequence[float]
                ) -> Sequence[int]:
                    return self._tile_code.get_feature_vector(parameters)

                def getFeatureVector(
                    self, parameters: Sequence[float]
                ) -> Sequence[int]:
                    return self.get_feature_vector(parameters)


class GradientDescentTileCode(
    GradientDescentTileCodeAbstract[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM]
):
    """
    Gradient Descent implementation.

    :tparam D: Number of dimension.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
    :tparam STATE_DIM: Number of dimension in State.
                       This also implies ACTION_DIM = D - STATE_DIM.
    """

    def __init__(
        self,
        tile_code: Any,
        step_size: float,
        discount_rate: float,
        lambda_: float,
        num_tilings: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(tile_code, step_size, discount_rate, lambda_, *args, **kwargs)
        self._tile_code = tile_code
        self._tileCode = tile_code
        self._step_size = step_size
        self._stepSize = step_size
        self._discount_rate = discount_rate
        self._discountRate = discount_rate
        self._lambda = lambda_

        if num_tilings is not None:
            self._num_tilings = num_tilings
        elif hasattr(tile_code, "get_num_tilings"):
            self._num_tilings = tile_code.get_num_tilings()
        elif hasattr(tile_code, "num_tilings"):
            self._num_tilings = tile_code.num_tilings
        elif hasattr(self, "_num_tilings"):
            pass
        else:
            self._num_tilings = None

    def update_weights(
        self,
        current_state_vector: Sequence[float],
        current_action_vector: Sequence[float],
        next_state_vector: Sequence[float],
        next_action_value: float,
        reward: float,
    ) -> None:
        current_params = concat_array(current_state_vector, current_action_vector)

        if hasattr(self, "get_feature_vector"):
            current_state_fv = self.get_feature_vector(current_params)
        else:
            current_state_fv = self.getFeatureVector(current_params)

        if hasattr(self, "get_value_from_feature_vector"):
            current_state_val = self.get_value_from_feature_vector(current_state_fv)
        else:
            current_state_val = self.getValueFromFeatureVector(current_state_fv)

        discount_rate = getattr(
            self, "_discount_rate", getattr(self, "_discountRate", 0.0)
        )
        td_error = (reward + discount_rate * next_action_value) - current_state_val

        step_size = getattr(self, "_step_size", getattr(self, "_stepSize", 0.0))

        num_tilings = getattr(self, "_num_tilings", None)
        if num_tilings is None:
            tile_code = getattr(self, "_tile_code", getattr(self, "_tileCode", None))
            if hasattr(tile_code, "get_num_tilings"):
                num_tilings = tile_code.get_num_tilings()
            elif hasattr(tile_code, "num_tilings"):
                num_tilings = tile_code.num_tilings
            else:
                num_tilings = 1

        tile_code = getattr(self, "_tile_code", getattr(self, "_tileCode", None))
        weight_update = td_error * (step_size / num_tilings)

        for f in current_state_fv:
            tile_code[f] += weight_update

    def updateWeights(
        self,
        currentStateVector: Sequence[float],
        currentActionVector: Sequence[float],
        nextStateVector: Sequence[float],
        nextActionValue: float,
        reward: float,
    ) -> None:
        self.update_weights(
            current_state_vector=currentStateVector,
            current_action_vector=currentActionVector,
            next_state_vector=nextStateVector,
            next_action_value=nextActionValue,
            reward=reward,
        )