"""Tabular algorithms solve the random walk (B -> A -> T) within 2 steps.

Ports Sarsa_test, SarsaET_test, QLearning_test, QLearningETWatkins_test,
DynaQ_test, DynaQETWatkins_test and DynaQPrioritizedSweeping_test.
"""
import pytest

from rl.agent import ActuatorFactory, Agent, SensorDiscreteFactory
from rl.algorithm import (
    DynaQETFactory,
    DynaQFactory,
    DynaQPrioritizedSweepingFactory,
    QLearningETFactory,
    QLearningFactory,
    SarsaETFactory,
    SarsaFactory,
)
from rl.policy import EpsilonGreedyFactory
from tests.helpers import B, L, R, T, RandomWalkEnvironmentFactory

ALGORITHMS = {
    "Sarsa": (lambda p: SarsaFactory(0.1, 0.9, p), 100),
    "SarsaET": (lambda p: SarsaETFactory(0.1, 0.9, p, 0.2), 100),
    "QLearning": (lambda p: QLearningFactory(0.1, 0.9, p), 100),
    "QLearningETWatkins": (lambda p: QLearningETFactory(0.1, 0.9, p, 0.5), 100),
    "DynaQ": (lambda p: DynaQFactory(0.1, 0.9, p, 100, 1.0, 1.0), 10),
    "DynaQETWatkins": (lambda p: DynaQETFactory(0.1, 0.9, p, 50, 1.0, 1.0, 0.5), 100),
    "DynaQPrioritizedSweeping": (
        lambda p: DynaQPrioritizedSweepingFactory(0.1, 0.9, p, 20, 1.0, 1.0, 0.1),
        100,
    ),
}


@pytest.mark.parametrize("name", ALGORITHMS)
def test_converges_on_random_walk(name):
    make_factory, episodes = ALGORITHMS[name]
    arw = ActuatorFactory({L, R}).get()
    srw = SensorDiscreteFactory(B).get()
    srw.addTerminalState(T)
    rwe = RandomWalkEnvironmentFactory(arw, srw).get()
    policy = EpsilonGreedyFactory(1.0).get()
    agent = Agent(rwe, make_factory(policy).get())

    iterationCount = 0
    for _ in range(episodes):
        agent.reset()  # Back to state B.
        iterationCount = agent.executeEpisode()
    assert iterationCount <= 2
