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

import pytest
from rl.agent import StateAction, StateActionPairValueComparison
from tests.helpers import RandomWalkEnvironment


def test_state_action_pair_value_comparison():
    b = RandomWalkEnvironment.State.B
    l = RandomWalkEnvironment.Action.L
    a = RandomWalkEnvironment.State.A

    sapvc = StateActionPairValueComparison()
    sav1 = (StateAction(b, l), 1.0)
    sav2 = (StateAction(a, l), 2.0)

    assert sapvc(sav2, sav1)