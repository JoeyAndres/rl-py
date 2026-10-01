from tests.helpers import A, B, L


def test_sensor_tracks_last_observed_state(random_walk):
    # SensorDiscrete_test.cpp in the C++ suite is identical to this one.
    agent, srw = random_walk
    assert srw.getLastObservedState() == B
    agent.applyAction(L)
    assert srw.getLastObservedState() == A
