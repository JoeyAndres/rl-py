from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, Generic, Set, TypeVar

S = TypeVar("S")
A = TypeVar("A")

from ..agent.StateAction import StateAction
from ..agent.StateActionTransition import StateActionTransition


class DynaQBase(Generic[S, A], ABC):
    """
    Abstract base class for DynaQ algorithms.

    DynaQ algorithm is a QLearning algorithm that employs a model
    to "model" the environment. Thus, in every update, the algorithm can
    do n simulations, enabling faster convergence to "optimal" policy.
    """

    def __init__(
        self,
        simulation_iteration_count: int,
        state_transition_greediness: float,
        state_transition_step_size: float,
    ) -> None:
        """
        :param simulation_iteration_count: number of simulations per update/backup.
        :param state_transition_greediness: greediness in selecting highest value model.
        :param state_transition_step_size: how fast does a model update a value of a
                                           state-action pair.
        """
        self._simulation_iteration_count: int = simulation_iteration_count
        self._state_action_transition_greediness: float = state_transition_greediness
        self._state_action_transition_step_size: float = state_transition_step_size
        self._model: Dict[StateAction[S, A], StateActionTransition[S]] = {}

    @property
    def _simulationIterationCount(self) -> int:
        return self._simulation_iteration_count

    @_simulationIterationCount.setter
    def _simulationIterationCount(self, value: int) -> None:
        self._simulation_iteration_count = value

    @property
    def _stateActionTransitionGreediness(self) -> float:
        return self._state_action_transition_greediness

    @_stateActionTransitionGreediness.setter
    def _stateActionTransitionGreediness(self, value: float) -> None:
        self._state_action_transition_greediness = value

    @property
    def _stateActionTransitionStepSize(self) -> float:
        return self._state_action_transition_step_size

    @_stateActionTransitionStepSize.setter
    def _stateActionTransitionStepSize(self, value: float) -> None:
        self._state_action_transition_step_size = value

    @abstractmethod
    def backUpStateActionPair(
        self,
        current_state_action: StateAction[S, A],
        reward: float,
        next_state_action_pair: StateAction[S, A],
    ) -> None:
        """
        Note that this is just a place-holder, since this will be aggregate
        backup implementations in children.
        Q[S, A] <- Q[S, A] + alpha * [R + gamma * max_{A'} Q[S', A'] - Q[S, A]]

        :param current_state_action: (S, A), current state-action pair.
        :param reward: R, reward after (S, A).
        :param next_state_action_pair: (S', A'), next state-action pair.
        """
        pass

    @abstractmethod
    def argMax(
        self,
        state: S,
        action_set: Set[A],
    ) -> A:
        """
        Note that this is just a place-holder, since this will be aggregate
        argMax implementations in children.
        Returns the action that will most "likely" gives the highest reward from the
        current state.

        :param state: the state to apply the argMax to.
        :param action_set: a set of possible actions.
        :return: the action that will "likely" gives the highest reward.
        """
        pass

    def _updateModel(
        self,
        current_state_action: StateAction[S, A],
        next_state: S,
        reward: float,
    ) -> None:
        """
        Update the stateAction map.

        :param current_state_action: currentStateAction agent is in.
        :param next_state: nextState by taking currentAction in currentState.
        :param reward: reward of currentStateAction.
        """
        self._addModel(current_state_action)
        st: StateActionTransition[S] = self._model[current_state_action]
        st.update(next_state, reward)

    def _addModelSafe(self, current_state_action: StateAction[S, A]) -> None:
        """
        Adds new stateAction pair to the model with mutex lock.

        :param current_state_action: state-action pair to be added.
        """
        self._addModel(current_state_action)

    def _addModel(self, current_state_action: StateAction[S, A]) -> None:
        """
        Adds new stateAction pair to the model without mutex lock.

        :param current_state_action: state-action pair to be added.
        """
        if current_state_action not in self._model:
            self._model[current_state_action] = StateActionTransition[S](
                self._state_action_transition_greediness,
                self._state_action_transition_step_size,
            )

    def _simulate(self, action_set: Set[A]) -> None:
        """
        Performs simulation _simulationIterationCount times.

        :param action_set: set of actions of agent.
        """
        if not self._model:
            return

        model_keys = list(self._model.keys())
        for _ in range(self._simulation_iteration_count):
            # Pick a random state-action in _model.
            item_key = random.choice(model_keys)
            st = self._model[item_key]

            trans_state = st.getNextState()
            next_action = self.argMax(trans_state, action_set)

            self.backUpStateActionPair(
                item_key,
                st.getReward(trans_state),
                StateAction(trans_state, next_action),
            )

    # Snake_case aliases for Pythonic method invocation
    back_up_state_action_pair = backUpStateActionPair
    arg_max = argMax
    _update_model = _updateModel
    _add_model_safe = _addModelSafe
    _add_model = _addModel