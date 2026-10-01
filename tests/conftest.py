import random

import pytest


@pytest.fixture(autouse=True)
def _seed_rng():
    # The C++ suite relied on an unseeded rand(); seed here for reproducibility.
    random.seed(0)
