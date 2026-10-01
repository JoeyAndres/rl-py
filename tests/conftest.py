import random

import pytest


@pytest.fixture(autouse=True)
def _seed_rng(request):
    # The C++ suite relied on an unseeded rand(); seed here for reproducibility.
    # Each rerun (pytest-rerunfailures) gets a new seed so a retry isn't an
    # identical replay of the failure.
    random.seed(getattr(request.node, "execution_count", 1) - 1)
