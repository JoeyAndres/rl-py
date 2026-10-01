# See Sensor_test.cpp. Does the same thing.

import pytest
from rl.agent import ActuatorFactory, SensorDiscreteFactory, Agent
from rl.algorithm import DynaQFactory
from rl.policy import EpsilonGreedyFactory
from tests.helpers import (
    RandomWalkEnvironmentFactory,
    A,
    B,
    L,
    R,
    T,
)


def test_sensor_discrete():
    # Setup actuator with actions.
    arw = ActuatorFactory([L, R]).get()
    # Setup sensor.
    srw = SensorDiscreteFactory(B).get()
    srw.addTerminalState(T)  # Setup terminal state.

    # Setup environment.
    rwe = RandomWalkEnvironmentFactory(arw, srw).get()

    policy = EpsilonGreedyFactory(1.0).get()
    dyna_q_algorithm = DynaQFactory(0.1, 0.9, policy, 100, 1.0, 1.0).get()
    agent = Agent(rwe, dyna_q_