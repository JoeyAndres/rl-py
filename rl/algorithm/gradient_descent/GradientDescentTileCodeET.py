from __future__ import annotations

from typing import (
    Any,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")
STATE_DIM = TypeVar("STATE_DIM")

FEATURE_VECTOR = Sequence[int]


def concat_array(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return list(a) + list(b)


concatArray = concat_array


from .GradientDescentTileCodeAbstract import GradientDescentTileCodeAbstract


class GradientDescentTileCodeET(
    GradientDescentTileCodeAbstract[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM],
    Generic[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM],
):
    """
    Gradient Descent eligibility traces.
    """

    _e: List[float]

    def __init__(
        self,
        tile_code: Any = None,
        step_size: float = 0.0,
        discount_rate: float = 0.0,
        lambda_: float = 0.0,
        num_tilings: Optional[int] = None,
        *,
        tileCode: Any = None,
        stepSize: Optional[float] = None,
        discountRate: Optional[float] = None,
        lambda_val: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        :param tile_code: Type of tile coding.
        :param step_size: Step size for gradient descent.
        :param discount_rate: Discount rate for gradient descent.
        :param lambda_: How influential is current state-action to their state-action.
        :param num_tilings: Number of tilings (optional if inferred from tile_code).
        """
        actual_tile_code = tileCode if tileCode is not None else tile_code
        actual_step_size = stepSize if stepSize is not None else step_size
        actual_discount_rate = discountRate if discountRate is not None else discount_rate
        actual_lambda = lambda_val if lambda_val is not None else lambda_

        super().__init__(
            actual_tile_code, actual_step_size, actual_discount_rate, actual_lambda
        )

        self._tile_code = actual_tile_code
        self._course_code = actual_tile_code
        self._step_size = actual_step_size
        self._discount_rate = actual_discount_rate
        self._lambda = actual_lambda
        self._discount_rate_times_lambda = actual_discount_rate * actual_lambda

        self._tileCode = self._tile_code
        self._courseCode = self._course_code
        self._stepSize = self._step_size
        self._discountRate = self._discount_rate
        self._discountRateTimesLambda = self._discount_rate_times_lambda

        if num_tilings is not None:
            self._num_tilings = int(num_tilings)
        elif hasattr(actual_tile_code, "get_num_tilings"):
            self._num_tilings = int(actual_tile_code.get_num_tilings())
        elif hasattr(actual_tile_code, "getNumTilings"):
            self._num_tilings = int(actual_tile_code.getNumTilings())
        elif hasattr(self, "NUM_TILINGS") and isinstance(self.NUM_TILINGS, int):
            self._num_tilings = self.NUM_TILINGS
        else:
            self._num_tilings = 1

        size = self.get_size()
        self._e = [0.0] * size

    def get_size(self) -> int:
        if hasattr(self, "_course_code") and hasattr(self._course_code, "get_size"):
            return self._course_code.get_size()
        if hasattr(self, "_course_code") and hasattr(self._course_code, "getSize"):
            return self._course_code.getSize()
        if hasattr(super(), "get_size"):
            return super().get_size()
        if hasattr(super(), "getSize"):
            return super().getSize()
        if hasattr(self, "_tile_code") and hasattr(self._tile_code, "__len__"):
            return len(self._tile_code)
        return len(self._e) if hasattr(self, "_e") else 0

    getSize = get_size

    def get_feature_vector(self, parameters: Sequence[float]) -> FEATURE_VECTOR:
        if hasattr(super(), "get_feature_vector"):
            return super().get_feature_vector(parameters)
        if hasattr(super(), "getFeatureVector"):
            return super().getFeatureVector(parameters)
        if hasattr(self._tile_code, "get_feature_vector"):
            return self._tile_code.get_feature_vector(parameters)
        if hasattr(self._tile_code, "getFeatureVector"):
            return self._tile_code.getFeatureVector(parameters)
        raise NotImplementedError("get_feature_vector is not implemented")

    getFeatureVector = get_feature_vector

    def get_value_from_feature_vector(self, fv: FEATURE_VECTOR) -> float:
        if hasattr(super(), "get_value_from_feature_vector"):
            return super().get_value_from_feature_vector(fv)
        if hasattr(super(), "getValueFromFeatureVector"):
            return super().getValueFromFeatureVector(fv)
        if hasattr(self._tile_code, "get_value_from_feature_vector"):
            return self._tile_code.get_value_from_feature_vector(fv)
        if hasattr(self._tile_code, "getValueFromFeatureVector"):
            return self._tile_code.getValueFromFeatureVector(fv)
        raise NotImplementedError("get_value_from_feature_vector is not implemented")

    getValueFromFeatureVector = get_value_from_feature_vector

    def increment_eligibility_traces(self, fv: Iterable[int]) -> None:
        """
        Increase the eligibility traces of a given feature vector.
        :param fv: Feature vector.
        """
        for f in fv:
            self._e[f] += 1.0

    incrementEligibilityTraces = increment_eligibility_traces

    def replace_eligibility_traces(self, fv: Iterable[int]) -> None:
        """
        Replace the eligibility traces for each feature vector by 1.0.
        :param fv: Feature vector.
        """
        for f in fv:
            self._e[f] = 1.0

    replaceEligibilityTraces = replace_eligibility_traces

    def decrease_eligibility_traces(self) -> None:
        """
        Decrease each eligibility trace by discount rate and lambda.
        """
        n = self.get_size()
        factor = self._discount_rate_times_lambda
        for i in range(n):
            self._e[i] *= factor

    decreaseEligibilityTraces = decrease_eligibility_traces

    def reset_eligibility_traces(self) -> None:
        """
        Set all eligibility traces to 0.0.
        """
        n = self.get_size()
        self._e = [0.0] * n

    resetEligibilityTraces = reset_eligibility_traces

    def back_up_weights(self, td_error: float) -> None:
        """
        Update weights with TD error.
        :param td_error: Temporal difference error.
        """
        multiplier = (self._step_size / self._num_tilings) * td_error
        n = self.get_size()
        for i in range(n - 1):
            try:
                self._tile_code[i] += multiplier * self._e[i]
            except (TypeError, KeyError):
                if hasattr(self._tile_code, "at"):
                    if hasattr(self._tile_code, "set_at"):
                        self._tile_code.set_at(i, self._tile_code.at(i) + multiplier * self._e[i])
                    elif hasattr(self._tile_code, "set"):
                        self._tile_code.set(i, self._tile_code.at(i) + multiplier * self._e[i])

    backUpWeights = back_up_weights

    def update_weights(
        self,
        current_state_vector: Sequence[float],
        current_action_vector: Sequence[float],
        next_state_vector: Sequence[float],
        next_action_value: float,
        reward: float,
    ) -> None:
        """
        :param current_state_vector: Array/sequence of current states.
        :param current_action_vector: Action taken to get to next_state_vector.
        :param next_state_vector: Array/sequence of next states.
        :param next_action_value: Value of next action.
        :param reward: Reward for taking next action.
        """
        current_params = concat_array(current_state_vector, current_action_vector)

        current_state_fv = self.get_feature_vector(current_params)
        self.increment_eligibility_traces(current_state_fv)

        td_error = (
            reward
            + self._discount_rate * next_action_value
            - self.get_value_from_feature_vector(current_state_fv)
        )

        self.back_up_weights(td_error)
        self.decrease_eligibility_traces()

    updateWeights = update_weights