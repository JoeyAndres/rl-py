"""Ports TileCode_test (TileCodeCorrect) and the hashed tile code tests."""
import pytest

from rl.coding import (
    DimensionInfo,
    TileCodeCorrect,
    TileCodeCorrectFactory,
    TileCodeMt1993764,
    TileCodeMt1993764Factory,
    TileCodeMurMur,
    TileCodeMurMurFactory,
    TileCodeSuperFastHash,
    TileCodeSuperFastHashFactory,
    TileCodeUNH,
    TileCodeUNHFactory,
)


@pytest.fixture
def dims():
    return [DimensionInfo(-0.5, 0.5, 3, 0), DimensionInfo(-0.5, 0.5, 3, 0)]


class TestTileCodeCorrect:
    @pytest.fixture
    def tileCode(self, dims):
        return TileCodeCorrect(dims, 4)

    def test_dimension(self, tileCode):
        assert tileCode.getDimension() == 2

    def test_size(self, tileCode):
        # (3+1) * (3+1) grid points * 4 tilings.
        assert tileCode.getSize() == 16 * 4

    def test_param_to_grid_value(self, tileCode):
        assert tileCode.paramToGridValue(-0.4, 1, 1) == 0

    def test_feature_vector(self, tileCode):
        assert tileCode.getFeatureVector([-0.4, -0.4]) == [0, 16, 32, 48]

    def test_factory(self, dims):
        assert isinstance(TileCodeCorrectFactory(dims, 4).get(), TileCodeCorrect)


HASHED = [
    (TileCodeMt1993764, TileCodeMt1993764Factory),
    (TileCodeMurMur, TileCodeMurMurFactory),
    (TileCodeSuperFastHash, TileCodeSuperFastHashFactory),
    (TileCodeUNH, TileCodeUNHFactory),
]


@pytest.mark.parametrize("cls,factory", HASHED, ids=lambda c: c.__name__)
class TestTileCodeHashed:
    def test_dimension(self, dims, cls, factory):
        assert cls(dims, 4, 100).getDimension() == 2

    def test_size_uses_size_hint(self, dims, cls, factory):
        assert cls(dims, 4, 100).getSize() == 100

    # The C++ tests don't predict hash values; additionally check the feature
    # vector is well-formed and deterministic.
    def test_feature_vector_in_range_and_deterministic(self, dims, cls, factory):
        tileCode = cls(dims, 4, 100)
        fv = tileCode.getFeatureVector([-0.4, 0.3])
        assert len(fv) == 4
        assert all(0 <= f < 100 for f in fv)
        assert tileCode.getFeatureVector([-0.4, 0.3]) == fv

    def test_factory(self, dims, cls, factory):
        tileCode = factory(dims, 4, 100).get()
        assert isinstance(tileCode, cls)
        assert tileCode.getSize() == 100
