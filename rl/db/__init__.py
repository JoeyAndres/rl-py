"""Database (Cassandra) persistence layer, mirroring the C++ ``rl::db`` namespace.

The module-level ``session`` attribute is kept in sync with
:func:`rl.db.utility.initialize` / :func:`rl.db.utility.terminate`.
"""

import sys as _sys

from . import utility as _utility
from .definition import RLKeySpace, TileCodeContainer, TileCodeContainerSegment
from .manipulation import InsertTileCodeContainer, InsertTileCodeContainerSegment
from .utility import (
    createKeySpace,
    executeStatement,
    genUuid,
    isInitialized,
)


def initialize() -> None:
    _utility.initialize()


def terminate() -> None:
    _utility.terminate()


def __getattr__(name):
    # Live views of connection state owned by rl.db.utility.
    if name in ("session", "cluster", "connectFuture"):
        return getattr(_utility, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


execute_statement = executeStatement
create_keyspace = createKeySpace
gen_uuid = genUuid
is_initialized = isInitialized
