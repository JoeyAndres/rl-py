# rl-py

Python port of the C++ reinforcement learning library [`rl`](../rl).

The class hierarchy mirrors the C++ one (see `tests/test_class_hierarchy.py`):
abstract bases (`Sensor`, `Environment`, `Policy`, `LearningAlgorithm`,
`TileCode`, ...) are `abc.ABC`s, and every concrete type has a matching
`*Factory`. Method names keep the C++ camelCase.

## C++ → Python mapping

| C++ | Python |
| --- | --- |
| `spState<S>` / `spAction<A>` (`shared_ptr`) | the value itself (Python references are shared) |
| `floatArray<N>` state/action in gradient descent | `tuple` of floats (hashable, used as dict keys) |
| `TileCode<D, NUM_TILINGS, WEIGHT_CONT>` | `TileCode(dimensionalInfos, numTilings, sizeHint=None, weightCont=None)`; `D = len(dimensionalInfos)` |
| `ReinforcementLearning<S, A, SAP_CONTAINER>` | `ReinforcementLearning(..., stateActionPairContainer=None)` |
| template-template params (`TILE_CODE_CLASS`, `REINFORCEMENT_LEARNING_GD`) | `TILE_CODE_CLASS` / `ALGORITHM_CLASS` class attributes on factory subclasses |

## Example

```python
from rl.agent import ActuatorFactory, Agent, SensorDiscreteFactory
from rl.algorithm import SarsaFactory
from rl.policy import EpsilonGreedyFactory

actuator = ActuatorFactory({LEFT, RIGHT}).get()
sensor = SensorDiscreteFactory(START).get()
sensor.addTerminalState(GOAL)
environment = MyEnvironment(actuator, sensor)  # subclass of rl.agent.Environment
agent = Agent(environment, SarsaFactory(0.1, 0.9, EpsilonGreedyFactory(1.0).get()).get())

for _ in range(100):
    agent.reset()
    steps = agent.executeEpisode()
```

## Tests

```sh
python -m venv .venv && .venv/bin/pip install -e . pytest pytest-timeout pytest-rerunfailures
.venv/bin/pytest -m "not slow"   # ~1s
.venv/bin/pytest                 # includes mountain car convergence (~1.5 min)
```

The Cassandra-backed `TileCodeContainer` tests (`ENABLE_DB` in C++) are skipped
unless `RL_ENABLE_DB=1` is set and a Cassandra node runs on `127.0.0.1`
(`pip install cassandra-driver`).

## Known differences from the C++

- `std::default_random_engine` is implementation-defined; `TileCodeUNH` uses the
  libstdc++ engine (`minstd_rand0`) for its hash table.
- `TileCodeUNH` hashes its actual tile components into `[0, size)`. The C++
  passed `sizeCache` as the coordinate count (reading past the array) and
  always hashed into 4096 slots.
- `GradientDescentTileCodeET.backUpWeights` updates every weight; the C++ loop
  skipped the last one.
