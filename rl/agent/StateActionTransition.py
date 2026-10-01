from typing import TypeVar, Generic, Dict, Optional, Union
import random
import copy

S = TypeVar('S')


class StateActionTransitionException(Exception):
    """Exception raised for errors in the StateActionTransition."""
    pass


class StateActionTransition(Generic[S]):
    """
    Used for modeling environment transitions for a specific state-action pair.
    """

    def __init__(
        self,
        greedy: Union[float, 'StateActionTransition[S]'],
        step_size: Optional[float] = None
    ) -> None:
        """
        Constructs StateActionTransition. Supports copy-construction if 'greedy'
        is an instance of StateActionTransition.
        """
        if isinstance(greedy, StateActionTransition):
            sat = greedy
            self._greedy: float = sat._greedy
            self._step_size: float = sat._step_size
            self._state_action_transition_frequency: Dict[S, float] = copy.deepcopy(
                sat._state_action_transition_frequency
            )
            self._state_action_transition_reward: Dict[S, float] = copy.deepcopy(
                sat._state_action_transition_reward
            )
        else:
            self._greedy = float(greedy)
            self._step_size = float(step_size) if step_size is not None else 0.0
            self._state_action_transition_frequency = {}
            self._state_action_transition_reward = {}

    def update(self, next_state: S, reward: float) -> None:
        """
        Updates the frequency distribution and reward for the given next state.
        """
        if next_state not in self._state_action_transition_frequency:
            self._state_action_transition_frequency[next_state] = 0.0
        if next_state not in self._state_action_transition_reward:
            self._state_action_transition_reward[next_state] = reward

        # Update Frequency
        # -- Lower the value of all other frequencies
        for state in self._state_action_transition_frequency:
            if state != next_state:
                self._state_action_transition_frequency[state] += (
                    self._step_size * (0.0 - self._state_action_transition_frequency[state])
                )

        # -- Increase the value of the next frequency
        self._state_action_transition_frequency[next_state] += (
            self._step_size * (1.0 - self._state_action_transition_frequency[next_state])
        )

        # Update reward
        self._state_action_transition_reward[next_state] += (
            1.0 * (reward - self._state_action_transition_reward[next_state])
        )

        assert len(self._state_action_transition_frequency) == len(self._state_action_transition_reward)

    def getReward(self, state: S) -> float:
        """
        Given a state, returns its latest reward info.
        """
        if not self._findState(state):
            raise StateActionTransitionException("state not yet added.")
        return self._state_action_transition_reward[state]

    def _findState(self, state: S) -> bool:
        return state in self._state_action_transition_frequency

    def getNextState(self) -> S:
        """
        Returns the possible next state based on its value.
        """
        if len(self._state_action_transition_frequency) == 0:
            raise StateActionTransitionException("nextStates are empty.")

        random_number_random_selection = random.uniform(0.0, 1.0)
        states = list(self._state_action_transition_frequency.keys())

        # If random selection > greediness, return a random state
        if random_number_random_selection > self._greedy:
            return random.choice(states)

        freq_val_sum = sum(self._state_action_transition_frequency.values())
        random_number_weighted_selection = random.uniform(0.0, freq_val_sum)

        for state, current_freq in self._state_action_transition_frequency.items():
            if current_freq > random_number_weighted_selection:
                return state
            random_number_weighted_selection -= current_freq

        return states[-1]

    def getSize(self) -> int:
        return len(self._state_action_transition_frequency)

    def setStepSize(self, step_size: float) -> None:
        self._step_size = step_size

    def getStepSize(self) -> float:
        return self._step_size

    def setGreedy(self, greedy: float) -> None:
        self._greedy = greedy

    def getGreedy(self) -> float:
        return self._greedy

    # Pythonic snake_case aliases
    get_reward = getReward
    get_next_state = getNextState
    get_size = getSize
    set_step_size = setStepSize
    get_step_size = getStepSize
    set_greedy = setGreedy
    get_greedy = getGreedy