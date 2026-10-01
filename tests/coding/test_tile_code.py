import pytest
from rl.coding import DimensionInfo, TileCodeCorrect


@pytest.fixture
def tile_code():
    dimensional_info_vector = [
        DimensionInfo(-0.5, 0.5, 3, 0),
        DimensionInfo(-0.5, 0.5, 3, 0),
    ]
    return TileCodeCorrect(2, 4, dimensional_info_vector)


def test_tile_code_get_dimension(tile_code):
    assert tile_code.get_dimension() == 2


def test_tile_code_get_size(tile_code):
    assert tile_code.get_size() == 16 * 4


def test_tile_code_param_to_grid_value(tile_code):
    assert tile