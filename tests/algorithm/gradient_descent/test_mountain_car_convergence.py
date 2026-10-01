"""Gradient-descent algorithms solve mountain car within 100 steps.

Ports SarsaGD_test, SarsaETGD_test, QLearningGD_test and QLearningETGD_test.
The ENABLE_DB (Cassandra TileCodeContainer) variant is not ported.
"""
import pytest

from rl.agent import ActuatorFactory, AgentGD
from rl.algorithm import (
    QLearningETGDFactory,
    QLearningGDFactory,
    SarsaETGDFactory,
    SarsaGDFactory,
)
from rl.coding import DimensionInfo, TileCodeCorrectFactory
from rl.policy import EpsilonGreedyFactory
from tests.helpers import MountainCarEnvironmentFactory, SensorMountainCarFactory

REVERSE, NEUTRAL, FORWARD = (0.0,), (1.0,), (2.0,)

# (factory, grid count for position/velocity) as in each C++ test.
ALGORITHMS = {
    "SarsaGD": (SarsaGDFactory, 10),
    "SarsaETGD": (SarsaETGDFactory, 10),
    "QLearningGD": (QLearningGDFactory, 7),
    "QLearningETGD": (QLearningETGDFactory, 7),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", ALGORITHMS)
def test_converges_on_mountain_car(name):
    factory, gridCount = ALGORITHMS[name]
    amc = ActuatorFactory({REVERSE, NEUTRAL, FORWARD}).get()
    smc = SensorMountainCarFactory.create()
    mce = MountainCarEnvironmentFactory(amc, smc).get()
    policy = EpsilonGreedyFactory(1.0).get()
    dimensionalInfoVector = [
        DimensionInfo(-1.2, 0.5, gridCount),  # Position.
        DimensionInfo(-0.07, 0.07, gridCount),  # Velocity.
        DimensionInfo(0.0, 2.0, 3, 0.0),  # Action.
    ]
    tileCode = TileCodeCorrectFactory(dimensionalInfoVector, 8).get()
    algorithm = factory(tileCode, 0.1, 1.0, 0.9, policy).get()
    agent = AgentGD(mce, algorithm)

    iterationCount = 0
    for _ in range(1000):
        agent.reset()
        iterationCount = agent.executeEpisode()
    assert iterationCount <= 100
