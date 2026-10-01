from rl.policy import EpsilonGreedyFactory


# Not in the C++ suite; EpsilonGreedy is exercised there only through agents.
def test_fully_greedy_picks_max():
    policy = EpsilonGreedyFactory(1.0).get()
    actionValueMap = {1: 1.0, 2: 5.0, 3: -1.0}
    for _ in range(100):
        assert policy.getAction(actionValueMap, set(actionValueMap)) == 2


def test_never_greedy_explores():
    policy = EpsilonGreedyFactory(0.0).get()
    actionValueMap = {1: 1.0, 2: 5.0, 3: -1.0}
    seen = {policy.getAction(actionValueMap, set(actionValueMap)) for _ in range(200)}
    assert seen == {1, 2, 3}
