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
from rl.coding import DimensionInfo


def test_dimensional_info_represents_a_dimension_in_tile_code():
    # Given DimensionalInfo instance
    dim = DimensionInfo(-0.5, 0.5, 3, 0)

    # When DimensionalInfo::GetOffsets
    # Then Returns 1/3
    assert (dim.GetOffsets() - 1.0 / 3.0) < 0.001

    # When DimensionalInfo::GetGridCountReal
    # Then Returns 4
    assert dim.GetGridCountReal() == 4

    # When DimensionalInfo::GetGridCountIdeal
    # Then Returns 3
    assert dim.GetGridCountIdeal() == 3

    # When DimensionalInfo::GetRangeDifference
    # Then Returns 1
    assert (dim.GetRangeDifference() - 1.0) < 0.001

    # When DimensionalInfo::getGeneralizationScale
    # Then Returns 0
    assert (dim.getGeneralizationScale() - 0.0) < 0.001