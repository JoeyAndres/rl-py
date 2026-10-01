from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

S = TypeVar("S")
A = TypeVar("A")

FLOAT = float

try:
    from rl.agent.StateAction import StateAction
    from rl.agent.StateActionNotExistException import StateActionNotExistException
    from rl.agent.StateActionPairContainer import (
        StateActionPairContainer,
        StateActionPairContainerSimple,
    )
    from rl.policy.Policy import Policy, spPolicy
    from rl.algorithm.LearningAlgorithm import LearningAlgorithm
except ImportError:
    try:
        from ..agent.StateAction import StateAction
        from ..agent.StateActionNotExistException import StateActionNotExistException
        from ..agent.StateActionPairContainer import (
            StateActionPairContainer,
            StateActionPairContainerSimple,
        )
        from ..policy.Policy import Policy, spPolicy
        from .LearningAlgorithm import LearningAlgorithm
    except (ImportError, ValueError):
        try:
            from agent.StateAction import StateAction
            from agent.StateActionNotExistException import StateActionNotExistException
            from agent.StateActionPairContainer import (
                StateActionPairContainer,
                StateActionPairContainerSimple,
            )
            from policy.Policy import Policy, spPolicy
            from algorithm.LearningAlgorithm import LearningAlgorithm
        except (ImportError, ValueError):
            class StateAction(Generic[S, A]):
                def __init__(self, state: S, action: A) -> None:
                    self.state: S = state
                    self.action: A = action

                def __eq__(self, other: object) -> bool:
                    if not isinstance(other, StateAction):
                        return False
                    return self.state == other.state and self.action == other.action

                def __hash__(self) -> int:
                    return hash((self.state, self.action))

            class StateActionNotExistException(Exception):
                pass

            class StateActionPairContainer(Generic[S, A]):
                def __init__(self) -> None:
                    self._map: Dict[StateAction[S, A], float] = {}

                def __getitem__(self, item: StateAction[S, A]) -> float:
                    if item not in self._map:
                        raise StateActionNotExistException("State-Action pair does not exist.")
                    return self._map[item]

                def __setitem__(self, item: StateAction[S, A], value: float) -> None:
                    self._map[item] = value

                def addState(self, state: S, value: float, actionSet: Iterable[A]) -> None:
                    for a in actionSet:
                        sa = StateAction(state, a)
                        if sa not in self._map:
                            self._map[sa] = value

                def addStateAction(self, stateAction: StateAction[S, A], value: float) -> None:
                    if stateAction not in self._map:
                        self._map[stateAction] = value

                def getStateActionValue(self, stateAction: StateAction[S, A]) -> float:
                    return self[stateAction]

                def setStateActionValue(self, stateAction: StateAction[S, A], reward: float) -> None:
                    self._map[stateAction] = reward

            StateActionPairContainerSimple = StateActionPairContainer

            class Policy(Generic[S, A], ABC):
                @abstractmethod
                def getAction(self, actionValueMap: Mapping[A, float], actionSet: Iterable[A]) -> A:
                    pass

            spPolicy = Policy

            class LearningAlgorithm(Generic[S, A], ABC):
                def __init__(self, controlPolicy: Optional[Policy[S, A]] = None) -> None:
                    self._controlPolicy: Optional[Policy[S, A]] = controlPolicy
                    self._learningPolicy: Optional[Policy[S, A]] = controlPolicy
                    self._defaultStateActionValue: float = 0.0

                @abstractmethod
                def update(
                    self,
                    currentStateAction: StateAction[S, A],
                    nextState: S,
                    reward: float,
                    actionSet: Iterable[A],
                ) -> None:
                    pass

                @abstractmethod
                def getAction(self, state: S, actionSet: Iterable[A]) -> A:
                    pass

                @abstractmethod
                def getStateActionValue(self, stateAction: StateAction[S, A]) -> float:
                    pass

                def setControlPolicy(self, policy: Policy[S, A]) -> None:
                    self._controlPolicy = policy

                def getControlPolicy(self) -> Optional[Policy[S, A]]:
                    return self._controlPolicy

                def reset(self) -> None:
                    pass

                def getDefaultStateActionValue(self) -> float:
                    return self._defaultStateActionValue

                def setDefaultStateActionValue(self, defaultStateActionValue: float) -> None:
                    self._defaultStateActionValue = float(defaultStateActionValue)

                def setLearningPolicy(self, policy: Policy[S, A]) -> None:
                    self._learningPolicy = policy

                def getLearningPolicy(self) -> Optional[Policy[S, A]]:
                    return self._learningPolicy

                def _getLearningPolicyAction(
                    self, actionValueMap: Mapping[A, float], actionSet: Iterable[A]
                ) -> A:
                    if self._learningPolicy is not None:
                        return self._learningPolicy.getAction(actionValueMap, actionSet)
                    return next(iter(actionSet))


class ReinforcementLearning(LearningAlgorithm[S, A], Generic[S, A]):
    """
    Abstract class for all Reinforcement learning algorithms.

    All reinforcement algorithm inherit from this class.
    """

    def __init__(
        self,
        stepSize: float,
        discountRate: float,
        policy: Optional[Policy[S, A]] = None,
        stateActionPairContainer: Optional[Any] = None,
    ) -> None:
        """
        :param stepSize: range [0.0, 1.0]. High step size means faster learning, but
                         less precise convergence.
        :param discountRate: range [0.0, 1.0]. High discount rate means more
                             consideration of future events.
        :param policy: online policy, that is policy used for action selection.
        :param stateActionPairContainer: optional container instance or factory.
        """
        super().__init__(policy)
        self._stepSize: float = float(stepSize)
        self._discountRate: float = float(discountRate)
        if not hasattr(self, "_defaultStateActionValue"):
            self._defaultStateActionValue = 0.0

        if stateActionPairContainer is None:
            self._stateActionPairContainer = StateActionPairContainerSimple()
        elif isinstance(stateActionPairContainer, type):
            self._stateActionPairContainer = stateActionPairContainer()
        else:
            self._stateActionPairContainer = stateActionPairContainer

    def argMax(self, state: S, actionSet: Iterable[A]) -> Optional[A]:
        """
        Returns the action that will most "likely" gives the highest reward from the
        current state.
        :param state: the state to apply the argMax to.
        :param actionSet: a set of possible actions.
        :return: the action that will "likely" gives the highest reward.
        """
        action_iter = iter(actionSet)
        try:
            greedAct = next(action_iter)
        except StopIteration:
            return None

        currentValue = self._defaultStateActionValue
        try:
            currentValue = self._stateActionPairContainer[StateAction(state, greedAct)]
        except (StateActionNotExistException, KeyError):
            # Happens if state is a terminal state in which no actions applies.
            pass

        for action in actionSet:
            value = self._defaultStateActionValue
            try:
                value = self._stateActionPairContainer[StateAction(state, action)]
            except (StateActionNotExistException, KeyError):
                # Happens if state is a terminal state in which no actions applies.
                pass

            if currentValue < value:
                currentValue = value
                greedAct = action

        return greedAct

    def getDiscountRate(self) -> float:
        """
        :return: current discount rate.
        """
        return self._discountRate

    def setDiscountRate(self, discountRate: float) -> None:
        """
        :param discountRate:
        """
        self._discountRate = float(discountRate)

    def getStepSize(self) -> float:
        """
        :return: current stepSize.
        """
        return self._stepSize

    def setStepSize(self, stepSize: float) -> None:
        """
        :param stepSize: set the step size of Reinforcement Learning.
        """
        self._stepSize = float(stepSize)

    def backUpStateActionPair(
        self,
        currentStateAction: StateAction[S, A],
        reward: float,
        nextStateActionPair: StateAction[S, A],
    ) -> None:
        """
        Does the main back up for all Temporal difference:
        Q[S, A] <- Q[S, A] + alpha * [R + gamma * max_{A'}Q[S', A'] - Q[S, A] ]
        :param currentStateAction: (S, A), current state-action pair.
        :param reward: R, reward after (S, A).
        :param nextStateActionPair: (S', A'), next state-action pair.
        """
        # Next state-action value (nSAV) and current state-action value (cSAV).
        nSAV = self.getStateActionValue(nextStateActionPair)
        cSAV = self.getStateActionValue(currentStateAction)
        discountRate = self._discountRate

        self.setStateActionValue(
            currentStateAction,
            cSAV + self._stepSize * (reward + discountRate * nSAV - cSAV),
        )

    def getLearningAction(self, currentState: S, actionSet: Iterable[A]) -> A:
        """
        Get Action with respect to learning/offline policy.
        :param currentState: state to acquire state-action values from.
        :param actionSet: Set of actions.
        :return: Action with respect to learning/offline policy.
        """
        self._stateActionPairContainer.addState(
            currentState, self._defaultStateActionValue, actionSet
        )
        actionValueMap = self._buildActionValueMap(actionSet, currentState)
        return self._getLearningPolicyAction(actionValueMap, actionSet)

    def getStateActionValue(self, stateAction: StateAction[S, A]) -> float:
        """
        :param stateAction: to acquire a value of.
        :return: the value of state-action pair.
        """
        return self._stateActionPairContainer.getStateActionValue(stateAction)

    def setStateActionValue(self, stateAction: StateAction[S, A], reward: float) -> None:
        """
        Q[ S, A ] <- R
        :param stateAction: to retrieve value from.
        :param reward: to set to the new state-action value.
        """
        self._stateActionPairContainer.setStateActionValue(stateAction, reward)

    def getStateActionPairContainer(self) -> Any:
        """
        :return: state-action pair container.
        """
        return self._stateActionPairContainer

    def setStateActionPairContainer(self, stateActionPairContainer: Any) -> None:
        """
        :param stateActionPairContainer: set state-action pair container.
        """
        self._stateActionPairContainer = stateActionPairContainer

    def updateStateAction(
        self,
        currentStateAction: StateAction[S, A],
        nextStateAction: StateAction[S, A],
        reward: float,
    ) -> None:
        _ = reward
        self._stateActionPairContainer.addStateAction(
            currentStateAction, self._defaultStateActionValue
        )
        self._stateActionPairContainer.addStateAction(
            nextStateAction, self._defaultStateActionValue
        )

    def getAction(self, currentState: S, actionSet: Iterable[A]) -> A:
        self._stateActionPairContainer.addState(
            currentState, self._defaultStateActionValue, actionSet
        )
        actionValueMap = self._buildActionValueMap(actionSet, currentState)
        if self._controlPolicy is None:
            raise RuntimeError("Control policy is not set.")
        return self._controlPolicy.getAction(actionValueMap, actionSet)

    def _buildActionValueMap(
        self, actionSet: Iterable[A], currentState: S
    ) -> Dict[A, float]:
        actionValueMap: Dict[A, float] = {}
        for action in actionSet:
            actionValueMap[action] = self._stateActionPairContainer[
                StateAction(currentState, action)
            ]
        return actionValueMap

    # Aliases matching Python naming conventions
    arg_max = argMax
    get_discount_rate = getDiscountRate
    set_discount_rate = setDiscountRate
    get_step_size = getStepSize
    set_step_size = setStepSize
    back_up_state_action_pair = backUpStateActionPair
    get_learning_action = getLearningAction
    get_state_action_value = getStateActionValue
    set_state_action_value = setStateActionValue
    get_state_action_pair_container = getStateActionPairContainer
    set_state_action_pair_container = setStateActionPairContainer
    update_state_action = updateStateAction
    get_action = getAction
    _build_action_value_map = _buildActionValueMap


spReinforcementLearning = ReinforcementLearning