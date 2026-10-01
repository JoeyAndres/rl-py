import pytest

from rl.coding import DimensionInfo


@pytest.fixture
def dim():
    return DimensionInfo(-0.5, 0.5, 3, 0)


def test_offsets(dim):
    assert dim.GetOffsets() == pytest.approx(1.0 / 3.0)


def test_grid_count_real(dim):
    assert dim.GetGridCountReal() == 4


def test_grid_count_ideal(dim):
    assert dim.GetGridCountIdeal() == 3


def test_range_difference(dim):
    assert dim.GetRangeDifference() == pytest.approx(1.0)


def test_generalization_scale(dim):
    assert dim.getGeneralizationScale() == pytest.approx(0.0)
