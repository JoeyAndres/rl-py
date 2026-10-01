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
from rl.coding import DimensionInfo, TileCodeMurMur


def test_tile_code_murmur_retrieves_the_correct_feature_vector():
    # Given TileCodeMurMur instance
    dimensional_info_vector = [
        DimensionInfo(-0.5, 0.5, 3, 0),
        DimensionInfo(-0.5, 0.5, 3, 0),
    ]

    tile_code = TileCodeMur