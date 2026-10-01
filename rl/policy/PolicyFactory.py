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

from abc import ABC
from typing import Generic, TypeVar

try:
    from ..utility.FactoryAbstract import FactoryAbstract
    from .Policy import Policy
except (ImportError, ValueError):
    try:
        from rl.utility.FactoryAbstract import FactoryAbstract
        from rl.policy.Policy import Policy
    except ImportError:
        from FactoryAbstract import FactoryAbstract
        from Policy import Policy

S = TypeVar("S")
A = TypeVar("A")


class PolicyFactory(FactoryAbstract[Policy[S, A]], Generic[S, A], ABC):
    """Abstract factory for Policy.

    :tparam S: State data type.
    :tparam A: Action data type.
    """

    def __init__(self) -> None:
        super().__init__()