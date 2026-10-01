"""Cassandra-backed weight storage. Mirrors the C++ ENABLE_DB-only tests.

Run with RL_ENABLE_DB=1 and a Cassandra node on 127.0.0.1.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RL_ENABLE_DB"), reason="requires Cassandra (set RL_ENABLE_DB=1)"
)


@pytest.fixture
def tcc():
    from rl.coding import TileCodeContainer

    container = TileCodeContainer(1000, 0.0)
    yield container
    container.delete2()


def test_container_member_starts_at_zero(tcc):
    assert tcc[0] == 0.0


def test_container_member_is_modified(tcc):
    tcc[0] = 69.0
    assert tcc[0] == 69.0
    assert tcc[1] == 0.0  # Sanity check.


@pytest.mark.parametrize("index", [0, 23, 50, 99])
def test_cell_starts_at_zero(tcc, index):
    from rl.coding import TileCodeContainerCell

    assert TileCodeContainerCell(tcc.getID(), 0, index) == 0.0


@pytest.mark.parametrize("index,value", [(0, 69.0), (99, -37.0)])
def test_cell_set(tcc, index, value):
    from rl.coding import TileCodeContainerCell

    cell = TileCodeContainerCell(tcc.getID(), 0, index)
    cell.set(value)
    assert cell == value


def test_segment_element_exists(tcc):
    from rl.coding import TileCodeContainerSegment

    assert TileCodeContainerSegment(tcc.getID(), 0).at(0) == 0.0
