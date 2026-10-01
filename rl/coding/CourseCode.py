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

from abc import ABC, abstractmethod
from typing import Generic, Optional, Sequence, TypeVar, get_args

D = TypeVar("D")

FLOAT = float
floatArray = Sequence[FLOAT]


class CourseCode(ABC, Generic[D]):
    """
    Abstract class for course coding.

    :tparam D: Number of dimension.
    """

    def __init__(self, dimension: Optional[int] = None) -> None:
        """
        No-arg constructor.
        """
        self._dimension: Optional[int] = dimension

    def getDimension(self) -> int:
        """
        @return number of dimension.
        """
        if self._dimension is not None:
            return self._dimension
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            args = get_args(orig_class)
            if args and isinstance(args[0], int):
                self._dimension = args[0]
                return self._dimension
        for base in getattr(self.__class__, "__orig_bases__", []):
            args = get_args(base)
            if args and isinstance(args[0], int):
                self._dimension = args[0]
                return self._dimension
        raise NotImplementedError("Dimension is not specified.")

    def get_dimension(self) -> int:
        return self.getDimension()

    @abstractmethod
    def getValueFromParameters(self, parameters: Sequence[float]) -> float:
        """
        Get the value of the parameters in the real space.
        @param parameters
        @return corresponding value.
        """
        pass

    def get_value_from_parameters(self, parameters: Sequence[float]) -> float:
        return self.getValueFromParameters(parameters)

    @abstractmethod
    def getSize(self) -> int:
        """
        @return size of the weight vector.
        """
        pass

    def get_size(self) -> int:
        return self.getSize()


# Wraps CourseCode
spCourseCode = CourseCode