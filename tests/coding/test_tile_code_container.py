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

"""TileCodeContainer, encapsulates the weight storage in tile code."""

import pytest
import rl.db
from rl.coding import TileCodeContainer


def test_schema_created_when_db_initialized():
    # GIVEN Uninitialized underlying db
    # WHEN Created
    # THEN Schema is created
    rl.db.initialize()


def test_tile_code_container_created_in_db():
    # GIVEN Initialized underlying db
    # WHEN New TileCodeContainer instance is created with the id agnostic constructor
    tcc = TileCodeContainer("id1", 1000, 0.0)
    # THEN It is added in db
    tcc.delete2()


def test_tile_code_container_member_access():
    # GIVEN An inserted TileCodeContainer
    tcc = TileCodeContainer("id2", 1000, 0.0)

    # WHEN A member is accessed.
    # THEN It is first 0.0
    assert tcc[0] == 0.0


def test_tile_code_container_member_modification():
    # GIVEN An inserted TileCodeContainer
    tcc = TileCodeContainer("id2", 1000, 0.0)

    # WHEN A member is modified
    # THEN It is modified
    tcc[0] = 69.0
    assert tcc[0] == 69.0
    assert tcc[1] == 0.0  # Sanity check.


def test_tile_code_container_deletion():
    # GIVEN An inserted TileCodeContainer
    tcc = TileCodeContainer("id2", 1000, 0.0)

    # WHEN TileCodeContainer is deleted
    # THEN id disappears
    tcc.delete2()


def test_db_terminate():
    # GIVEN Initialized underlying db
    # WHEN Terminate
    # THEN Connection is ceased.
    rl.db.terminate()