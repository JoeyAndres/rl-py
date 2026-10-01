from rl.agent import StateAction


class TestStateAction:
    state01, action01 = 10, 10
    state02, action02 = 20, 20

    def test_accessors_retrieve_values(self):
        sa = StateAction(self.state01, self.action01)
        assert sa.getState() == self.state01
        assert sa.getAction() == self.action01

    def test_equality(self):
        sa1 = StateAction(self.state01, self.action01)
        sa2 = StateAction(self.state01, self.action01)
        sa3 = StateAction(self.state02, self.action01)
        assert sa1 == sa2
        assert sa1 != sa3
        assert sa2 != sa3

    def test_less_than_compares_lexically(self):
        sa1 = StateAction(self.state01, self.action01)
        sa2 = StateAction(self.state01, self.action02)
        sa3 = StateAction(self.state01, self.action02)
        sa4 = StateAction(self.state02, self.action01)
        assert sa1 != sa2
        assert sa1 < sa2
        assert sa3 != sa4
        assert sa3 < sa4
