from tests.helpers import A, B, L


def test_agent_interacts_with_environment(random_walk):
    agent, _ = random_walk
    assert agent.getLastObservedState() == B
    agent.applyAction(L)
    assert agent.getLastObservedState() == A
