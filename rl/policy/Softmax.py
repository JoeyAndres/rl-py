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

import math
import random
from abc import ABC, abstractmethod
from typing import Dict, Generic, Mapping, Optional, Set, TypeVar

S = TypeVar("S")
A = TypeVar("A")

try:
    from .Policy import Policy
except ImportError:
    try:
        from .policy import Policy
    except ImportError:
        try:
            from Policy import Policy  # type: ignore
        except ImportError:
            try:
                from policy import Policy  # type: ignore
            except ImportError:

                class Policy(Generic[S, A], ABC):  # type: ignore
                    @abstractmethod
                    def getAction(
                        self,
                        actionValues: Mapping[A, float],
                        actionSet: Set[A],
                        maxAction: Optional[A] = None,
                    ) -> A:
                        pass


class Softmax(Policy[S, A], Generic[S, A]):
    """Greedy action is given highest selection probability, but all others
    are ranked and weighted according to their value estimates.

    Greedy action is given highest selection probability, but all others
    are ranked and weighted according to their value estimates. In contrast with
    EpsilonGreedy algorithm where non-greedy actions have equal chance of being
    chosen, non-greedy actions in Softmax are chosen depending on their value
    estimates. The higher the value estimates, the more likely to be chosen.

    In essence, Softmax chooses action a with probability,
        e^(Q_t(a) / tau) / sum_{i=1}^n e^(Q_t(i) / tau)
    where tau is the temperature. High temperature causes the actions to be
    all (nearly) equiprobable. Low temperatures cause a greater difference in
    selection probability that differ in their action estimates.

    This policy algorithm is adapted from Sutton and Barto rl book 2nd edition pg 31.
    """

    def __init__(self, temperature: float) -> None:
        super().__init__()
        self._temperature: float = float(temperature)
        self._random: random.Random = random.Random()

    def getAction(
        self,
        actionValues: Mapping[A, float],
        actionSet: Set[A],
        maxAction: Optional[A] = None,
    ) -> A:
        # Acquire E(i=1...n) e^(Q(i)/temp)
        sum_exp: float = 0.0
        for action in actionSet:
            sum_exp += math.exp(actionValues[action] / self._temperature)

        # Acquire probability for each action.
        action_probability_map: Dict[A, float] = {}
        for action in actionSet:
            probability = math.exp(actionValues[action] / self._temperature) / sum_exp
            action_probability_map[action] = probability

        # Choose action based on probability.
        random_number_weighted_selection = self._random.random()

        # Get the one that matches the random value.
        for action, current_probability in action_probability_map.items():
            if current_probability > random_number_weighted_selection:
                return action
            random_number_weighted_selection -= current_probability

        if action_probability_map:
            return next(reversed(action_probability_map))

        assert False, "Softmax action selection failed"

    get_action = getAction


spSoftmax = Softmax