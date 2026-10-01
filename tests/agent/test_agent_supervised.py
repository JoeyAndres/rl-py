import pytest

from rl.agent import ActionContainerFactory, AgentSupervised, StateAction
from rl.algorithm import SarsaFactory
from rl.policy import EpsilonGreedyFactory


def test_supervised_agent_develop_an_accurate_model_of_the_environment():
    # GIVEN: A binary environment in which 1 is good and 0 is bad.
    state0 = 0
    state1 = 1

    action0 = 0
    action1 = 1

    policy = EpsilonGreedyFactory(1.0).get()
    sarsa_algorithm = SarsaFactory(0.1, 0.9, policy).get()
    action_set = ActionContainerFactory([action0, action1]).get()
    supervised_agent = AgentSupervised(action_set, sarsa_algorithm)

    # WHEN: When I train 1 to be good and 0 to be bad
    # We don't transition anywhere. It's just being in state 1 is good.
    supervised_agent.train(state1, action1, 1000, state1)

    # Same deal.
    supervised_agent.train(state0, action0, -1000, state0)

    # THEN: Agent should know that 1 should be good and 0 should be bad
    value1 = sarsa_algorithm.get_state_action_value(
        StateAction(state1, action1)
    )
    value0 = sarsa_algorithm.get_state_action_value(
        StateAction(state0, action0)
    )

    assert value1 > value0