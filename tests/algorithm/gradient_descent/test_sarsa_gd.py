import pytest
from rl.agent import ActuatorFactory, AgentGD
from rl.algorithm import SarsaGDFactory
from rl.coding import DimensionInfo, TileCodeCorrectFactory
from rl.policy import EpsilonGreedyFactory
from tests.helpers import MountainCarEnvironmentFactory, SensorMountainCarFactory


@pytest.mark.sarsagd
def test_sarsa_gradient_descent_converges_to_solution():
    # GIVEN("A Mountain Car environment")
    reverse = [0.0]
    neutral = [1.0]
    forward = [2.0]

    amc = ActuatorFactory([reverse, neutral, forward]).get()

    # Setup sensor.
    smc = SensorMountainCarFactory.create()
    # Setup environment.
    mce = MountainCarEnvironmentFactory(amc