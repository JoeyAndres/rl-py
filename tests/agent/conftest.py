import pytest

from rl.agent import ActuatorFactory, Agent, SensorDiscreteFactory
from rl.algorithm import DynaQFactory
from rl.policy import EpsilonGreedyFactory
from tests.helpers import B, L, R, T, RandomWalkEnvironmentFactory


@pytest.fixture
def random_walk():
    """Random walk environment wired to a DynaQ agent, starting at B."""
    arw = ActuatorFactory({L, R}).get()
    srw = SensorDiscreteFactory(B).get()
    srw.addTerminalState(T)
    rwe = RandomWalkEnvironmentFactory(arw, srw).get()
    policy = EpsilonGreedyFactory(1.0).get()
    dynaQ = DynaQFactory(0.1, 0.9, policy, 100, 1.0, 1.0).get()
    return Agent(rwe, dynaQ), srw
