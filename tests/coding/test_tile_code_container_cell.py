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
from rl.coding import TileCodeContainer, TileCodeContainerCell


@pytest.fixture
def tcc():
    """Fixture providing a default TileCodeContainer instance and ensuring cleanup."""
    container = TileCodeContainer("id1", 1000, 0.0)
    yield container
    container.delete2()


def test_tile_code_container_cell_initial_values(tcc):
    """
    Scenario: TileCodeContainerCell, a single weight in TileCodeContainer.
    Given: A default TileCodeContainerSegment instance
    When: TileCodeContainerCell is initialized with index 0, 23, 50, 99
    Then: All have 0 values
    """
    tccc1 = TileCodeContainerCell(tcc.getID(), 0, 0)
    tccc2 = TileCodeContainerCell(tcc.getID(), 0, 23)
    tccc3 = TileCodeContainerCell(tcc.getID(), 0, 50)
    tccc4 = TileCodeContainerCell(tcc.getID(), 0, 99)

    assert tccc1 == 0.0
    assert tccc2 == 0.0
    assert tccc3 == 0.0
    assert tccc4 == 0.0


def test_tile_code_container_cell_change_value_index_0(tcc):
    """
    Scenario: TileCodeContainerCell, a single weight in TileCodeContainer.
    Given: A TileCodeContainerCell instance
    When: I change the value of index 0 from 0.0 to 69.0
    Then: That index 0 cell is now 69.0
    """
    tccc = TileCodeContainerCell(tcc.getID(), 0, 0)
    tccc.set(69.0)

    assert tccc == 69.0


def test_tile_code_container_cell_change_value_index_99(tcc):
    """
    Scenario: TileCodeContainerCell, a single weight in TileCodeContainer.
    Given: A TileCodeContainerCell instance
    When: I change the value of index 99 from 0.0 to -37.0
    Then: That index 99 cell is now -37.0
    """
    tccc = TileCodeContainerCell(tcc.getID(), 0, 99)
    tccc.set(-37.0)

    assert tccc == -37.0