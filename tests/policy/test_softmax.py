from rl.policy import SoftmaxFactory


def test_selection_probability_increases_with_value():
    policy = SoftmaxFactory(0.9).get()
    actionValueMap = {1: 1.0, 2: 2.0, 3: -1.0, 4: 2.2}
    actionSet = set(actionValueMap)

    counts = {a: 0 for a in actionSet}
    for _ in range(1000):
        counts[policy.getAction(actionValueMap, actionSet)] += 1

    assert counts[1] > counts[3]
    assert counts[2] > counts[1]
    assert counts[4] > counts[2]
