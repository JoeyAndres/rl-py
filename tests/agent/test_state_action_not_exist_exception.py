"""
rl - Reinforcement Learning
Copyright (C) 2016  Joey Andres<yeojserdna@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import pytest

# Just testing simulated installed library.
from rl.agent import StateActionNotExistException


def test_state_action_not_exist_exception():
    # Just testing simulated installed library.
    assert issubclass(StateActionNotExistException, Exception)

    exception = StateActionNotExistException(" extra detail")
    assert "StateAction pair does not exist." in str(exception)
    assert " extra detail" in str(exception)

    with pytest.raises(StateActionNotExistException):
        raise StateActionNotExistException(" extra detail")