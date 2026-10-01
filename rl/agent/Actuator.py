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

from typing import Generic, Optional, Set, TypeVar

try:
    from .ActionContainer import ActionContainer
except ImportError:
    try:
        from ActionContainer import ActionContainer
    except ImportError:
        try:
            from .action_container import ActionContainer
        except ImportError:
            from action_container import ActionContainer

A = TypeVar("A")


class Actuator(ActionContainer[A]):
    """Base class and interface for all Actuator objects.

    Base class and interface for all actuator objects. One can override
    or extend virtual functions here to direct output to environment. For
    example, consider a line following robot example.
    """

    def __init__(self, action_set: Optional[Set[A]] = None) -> None:
        """Constructor for Actuator.

        :param action_set: Set of actions when known.
        """
        if action_set is not None:
            super().__init__(action_set)
        else:
            super().__init__()


# Type alias corresponding to C++ spActuator
spActuator = Actuator