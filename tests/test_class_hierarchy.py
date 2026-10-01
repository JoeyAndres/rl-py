"""The Python port keeps the C++ class hierarchy (subclass -> bases)."""
import inspect

import pytest

import rl
from rl import agent, algorithm, coding, hash, policy, utility

HIERARCHY = {
    # agent
    "Actuator": ["ActionContainer"],
    "ActionContainerFactory": ["FactoryAbstract"],
    "ActuatorFactory": ["FactoryAbstract"],
    "EnvironmentFactory": ["FactoryAbstract"],
    "SensorDiscrete": ["Sensor"],
    "SensorDiscreteFactory": ["FactoryAbstract"],
    "SensorFactory": ["FactoryAbstract"],
    "StateActionNotExistException": ["Exception"],
    "StateActionTransitionException": ["Exception"],
    "StateNotExistException": ["Exception"],
    # policy
    "EpsilonGreedy": ["Policy"],
    "Softmax": ["Policy"],
    "PolicyFactory": ["FactoryAbstract"],
    "EpsilonGreedyFactory": ["PolicyFactory"],
    "SoftmaxFactory": ["PolicyFactory"],
    # tabular algorithms
    "ReinforcementLearning": ["LearningAlgorithm"],
    "Sarsa": ["ReinforcementLearning"],
    "QLearning": ["ReinforcementLearning"],
    "SarsaET": ["Sarsa", "EligibilityTraces"],
    "QLearningET": ["QLearning", "EligibilityTraces"],
    "DynaQRLMP": ["ReinforcementLearning", "DynaQBase"],
    "DynaQ": ["DynaQRLMP"],
    "DynaQET": ["DynaQ", "EligibilityTraces"],
    "DynaQPrioritizedSweeping": ["DynaQ"],
    "LearningAlgorithmFactory": ["FactoryAbstract"],
    "ReinforcementLearningFactory": ["FactoryAbstract"],
    "SarsaFactory": ["ReinforcementLearningFactory"],
    "SarsaETFactory": ["ReinforcementLearningFactory"],
    "QLearningFactory": ["ReinforcementLearningFactory"],
    "QLearningETFactory": ["ReinforcementLearningFactory"],
    "DynaQFactory": ["ReinforcementLearningFactory"],
    "DynaQETFactory": ["ReinforcementLearningFactory"],
    "DynaQPrioritizedSweepingFactory": ["ReinforcementLearningFactory"],
    # gradient descent
    "GradientDescentTileCodeAbstract": ["GradientDescentAbstract"],
    "GradientDescentTileCode": ["GradientDescentTileCodeAbstract"],
    "GradientDescentTileCodeET": ["GradientDescentTileCodeAbstract"],
    "ReinforcementLearningGDAbstract": ["LearningAlgorithm"],
    "ReinforcementLearningGD": ["ReinforcementLearningGDAbstract"],
    "ReinforcementLearningGDET": ["ReinforcementLearningGDAbstract"],
    "SarsaGD": ["ReinforcementLearningGD"],
    "QLearningGD": ["ReinforcementLearningGD"],
    "SarsaETGD": ["ReinforcementLearningGDET"],
    "QLearningETGD": ["ReinforcementLearningGDET"],
    "ReinforcementLearningGDFactory": ["LearningAlgorithmFactory"],
    "SarsaGDFactory": ["ReinforcementLearningGDFactory"],
    "SarsaETGDFactory": ["ReinforcementLearningGDFactory"],
    "QLearningGDFactory": ["ReinforcementLearningGDFactory"],
    "QLearningETGDFactory": ["ReinforcementLearningGDFactory"],
    # coding
    "TileCode": ["CourseCode", "IndexAccessorInterface"],
    "TileCodeCorrect": ["TileCode"],
    "TileCodeMt1993764": ["TileCode"],
    "TileCodeMurMur": ["TileCode"],
    "TileCodeSuperFastHash": ["TileCode"],
    "TileCodeUNH": ["TileCode"],
    "TileCodeFactory": ["FactoryAbstract"],
    "TileCodeHashedFactory": ["TileCodeFactory"],
    "TileCodeCorrectFactory": ["TileCodeFactory"],
    "TileCodeMt1993764Factory": ["TileCodeHashedFactory"],
    "TileCodeMurMurFactory": ["TileCodeHashedFactory"],
    "TileCodeSuperFastHashFactory": ["TileCodeHashedFactory"],
    "TileCodeUNHFactory": ["TileCodeHashedFactory"],
    "TileCodeContainer": ["CRUDInterface"],
    "TileCodeContainerSegment": ["CRUDInterface", "IndexAccessorInterface"],
    # hash
    "Murmur3": ["HashInterface"],
    "SuperFastHash": ["HashInterface"],
    "UNH": ["HashInterface"],
}

ABSTRACT = [
    "Sensor", "Environment", "Policy", "LearningAlgorithm", "DynaQBase",
    "GradientDescentAbstract", "CourseCode", "TileCode", "HashInterface",
    "CRUDInterface", "IndexAccessorInterface",
]


def _lookup(name):
    if name == "Exception":
        return Exception
    for mod in (agent, algorithm, coding, hash, policy, utility):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise LookupError(name)


@pytest.mark.parametrize("cls,bases", HIERARCHY.items(), ids=list(HIERARCHY))
def test_subclass_relationship(cls, bases):
    sub = _lookup(cls)
    for base in bases:
        assert issubclass(sub, _lookup(base)), f"{cls} should derive from {base}"


@pytest.mark.parametrize("name", ABSTRACT)
def test_abstract_bases_cannot_be_instantiated(name):
    assert inspect.isabstract(_lookup(name)), f"{name} should be abstract"
