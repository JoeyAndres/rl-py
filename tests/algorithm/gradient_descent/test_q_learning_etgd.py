import pytest

from rl.agent import ActuatorFactory, AgentGD
from rl.algorithm import QLearningETGDFactory
from rl.coding import DimensionInfo, TileCodeCorrectFactory
from rl.policy import EpsilonGreedyFactory
from tests.helpers import MountainCarEnvironmentFactory, SensorMountainCarFactory


def test_q_learning_eligibility_traces_and_gradient_descent_converge_to_a_solution():
    """Scenario: Q-learning Eligibility Traces and Gradient Descent converge to a solution."""
    # Given A Mountain Car environment
    # Actions.
    reverse = [0.0]
    neutral = [1.0]
    forward = [2.0]

    amc = ActuatorFactory([reverse, neutral, forward]).get()

    # Setup sensor.
    smc = SensorMountainCarFactory.create()

    # Setup environment.
    mce = MountainCarEnvironmentFactory(amc, smc).get()

    policy = EpsilonGreedyFactory(1.0).get()

    # Setup tile coding.
    dimensional_info_vector = [
        DimensionInfo(-1.2, 0.5, 7),  # Velocity.
        DimensionInfo(-0.07, 0.07, 7),  # Position.
        # Action dimension.
        DimensionInfo(0.0, 2.0, 3, 0.0),
    ]

    # Setup tile coding with 10 offsets.
    tile_code = TileCodeCorrectFactory(dimensional_info_vector, 8).get()
    q_learning = QLearningETGDFactory(
        tile_code, 0.1, 1.0, 0.9, policy
    ).get()
    agent = AgentGD(mce, q_learning)

    # When We do multiple episodes
    iteration_count = 0
    for _ in range(1000):
        mce.reset()
        iteration_count = agent.execute_episode()

    # Then At the end, we solve the Mountain Car environment in 100 iteration
    assert iteration_count <= 100