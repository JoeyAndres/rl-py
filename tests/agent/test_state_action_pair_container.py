from rl.agent import StateAction, StateActionPairContainer


def test_add_state_action():
    container = StateActionPairContainer()
    container.addStateAction(StateAction(0, 1), 1)
    assert container.size() == 1


def test_add_state_with_actions():
    container = StateActionPairContainer()
    container.addState(0, -1, {1, 2, 3, 4, 5})
    assert container.size() == 5


def test_add_action_with_states():
    container = StateActionPairContainer()
    container.addAction(5, -1, {0, 1, 2, 3, 4})
    assert container.size() == 5


def test_set_state_action_value():
    container = StateActionPairContainer()
    container.addStateAction(StateAction(0, 1), 1)
    assert container.getStateActionValue(StateAction(0, 1)) == 1
    container.setStateActionValue(StateAction(0, 1), 2)
    assert container.getStateActionValue(StateAction(0, 1)) == 2
