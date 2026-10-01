from rl.agent import ActionContainer


def test_no_arg_container_is_empty():
    assert len(ActionContainer().getActionSet()) == 0


def test_container_initialized_with_actions():
    container = ActionContainer({1, 2, 3, 4})
    assert len(container.getActionSet()) == 4
