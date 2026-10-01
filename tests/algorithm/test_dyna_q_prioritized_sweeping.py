import pytest

from rl.agent import ActuatorFactory, Agent, SensorDiscreteFactory
from rl.algorithm import DynaQPrioritizedSweepingFactory
from rl.policy import EpsilonGreedyFactory
from tests.helpers import (
    B,
    L,
    R,
    RandomWalkEnvironmentFactory,
    T,
)


def test_dyna_q_prioritized_sweeping_converge_to_a_solution():
    # Setup actuator with actions.
    arw = ActuatorFactory([L, R]).get()
    # Setup sensor.
    srw = SensorDiscreteFactory(B).get()
    srw.add_terminal_state(T)  # Setup terminal state.

    # Setup environment.
    rwe = RandomWalkEnvironmentFactory(arw, srw).get()

    policy = EpsilonGreedyFactory(1.0).get()
    dyna_q_prioritized_sweeping = DynaQPrioritizedSweepingFactory(
        0.1, 0.9, policy, 20, 1.0, 1.0, 0.1
    ).get()

    agent = Agent(rwe, dyna_q_prioritized_sweeping)

    iteration_count = 0
    for _ in range(100):
        agent.reset()  # Overloaded to go back to set b.

        iteration_count = agent.execute_episode()

        assert iteration_count <= 2