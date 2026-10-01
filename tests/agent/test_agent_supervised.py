from rl.agent import ActionContainerFactory, AgentSupervised, StateAction
from rl.algorithm import SarsaFactory
from rl.policy import EpsilonGreedyFactory


def test_supervised_agent_learns_model():
    # Environment_test.cpp in the C++ suite is a copy of this scenario.
    state0, state1 = 0, 1
    action0, action1 = 0, 1
    policy = EpsilonGreedyFactory(1.0).get()
    sarsa = SarsaFactory(0.1, 0.9, policy).get()
    actionSet = ActionContainerFactory({action0, action1}).get()
    agent = AgentSupervised(actionSet, sarsa)

    # We don't transition anywhere; just being in state 1 is good.
    agent.train(state1, action1, 1000, state1)
    agent.train(state0, action0, -1000, state0)

    value1 = sarsa.getStateActionValue(StateAction(state1, action1))
    value0 = sarsa.getStateActionValue(StateAction(state0, action0))
    assert value1 > value0
