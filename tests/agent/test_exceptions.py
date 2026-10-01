import pytest

from rl.agent import (
    StateActionNotExistException,
    StateActionTransitionException,
    StateNotExistException,
)


# The C++ suite only checks these link from the installed library; here we also
# check they are raisable exceptions carrying their message.
@pytest.mark.parametrize(
    "exc",
    [StateActionNotExistException, StateActionTransitionException, StateNotExistException],
)
def test_exception_is_raisable(exc):
    with pytest.raises(exc, match="boom"):
        raise exc("boom")
