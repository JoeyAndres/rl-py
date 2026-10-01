from rl.agent import StateAction, StateActionPairValueComparison
from tests.helpers import A, B, L


def test_compares_state_action_pairs():
    sapvc = StateActionPairValueComparison()
    sav1 = (StateAction(B, L), 1.0)
    sav2 = (StateAction(A, L), 2.0)
    assert sapvc(sav2, sav1)
