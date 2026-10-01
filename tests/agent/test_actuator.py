from rl.agent import ActuatorFactory
from tests.helpers import L, R


def test_actuator_contains_actions():
    actuator = ActuatorFactory({L, R}).get()
    assert len(actuator.getActionSet()) == 2
