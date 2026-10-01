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

from __future__ import annotations
import sys
import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

try:
    from cassandra.cluster import Cluster, Session, ResponseFuture
except ImportError:
    Cluster = None
    Session = None
    ResponseFuture = None

RLKeySpace: str = (
    "CREATE KEYSPACE IF NOT EXISTS rl\n"
    "  WITH replication = {'class': 'SimpleStrategy', 'replication_factor' : 3};"
)


class DatabaseUtilityInterface(ABC):
    """Abstract interface defining database utility operations."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize connections and schema."""
        pass

    @abstractmethod
    def terminate(self) -> None:
        """Terminate connections."""
        pass

    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if connections are initialized."""
        pass

    @abstractmethod
    def execute_statement(self, stmt_str: str) -> Optional[Any]:
        """Execute a CQL statement and return its result."""
        pass

    @abstractmethod
    def create_keyspace(self) -> None:
        """Create the keyspace to be used by this application."""
        pass

    @abstractmethod
    def gen_uuid(self) -> uuid.UUID:
        """Generate a UUID."""
        pass


class DBUtility(DatabaseUtilityInterface):
    """
    Implementation of database utility operations managing Cassandra
    cluster and session connections.
    """

    _instance: Optional[DBUtility] = None

    def __init__(self, contact_points: Sequence[str] = ("127.0.0.1",)) -> None:
        self._cluster: Optional[Any] = None
        self._session: Optional[Any] = None
        self._connect_future: Optional[Any] = None
        self._contact_points: list[str] = list(contact_points)

    @classmethod
    def get_instance(cls) -> DBUtility:
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def cluster(self) -> Optional[Any]:
        return self._cluster

    @cluster.setter
    def cluster(self, value: Optional[Any]) -> None:
        self._cluster = value

    @property
    def session(self) -> Optional[Any]:
        return self._session

    @session.setter
    def session(self, value: Optional[Any]) -> None:
        self._session = value

    @property
    def connect_future(self) -> Optional[Any]:
        return self._connect_future

    @connect_future.setter
    def connect_future(self, value: Optional[Any]) -> None:
        self._connect_future = value

    @property
    def connectFuture(self) -> Optional[Any]:
        return self._connect_future

    @connectFuture.setter
    def connectFuture(self, value: Optional[Any]) -> None:
        self._connect_future = value

    def is_initialized(self) -> bool:
        return (
            self._connect_future is not None
            and self._session is not None
            and self._cluster is not None
        )

    def isInitialized(self) -> bool:
        return self.is_initialized()

    def initialize(self) -> None:
        if self.is_initialized():
            return

        try:
            if Cluster is None:
                raise RuntimeError("Cassandra driver (cassandra-driver) is not available.")

            self._cluster = Cluster(contact_points=self._contact_points)
            self._connect_future = self._cluster.connect_async()
            self._session = self._connect_future.result()
        except Exception as e:
            self.terminate()
            sys.stderr.write(f"Connect result: {e}\n")
            return

        self.create_keyspace()

    def terminate(self) -> None:
        if not self.is_initialized():
            return

        if self._session is not None:
            try:
                self._session.shutdown()
            except Exception:
                pass
            self._session = None

        if self._cluster is not None:
            try:
                self._cluster.shutdown()
            except Exception:
                pass
            self._cluster = None

        self._connect_future = None

    def execute_statement(self, stmt_str: str) -> Optional[Any]:
        if self._session is None:
            sys.stderr.write("Session is not initialized.\n")
            return None

        try:
            return self._session.execute(stmt_str)
        except Exception as e:
            sys.stderr.write(f"{e}\n")
            return None

    def executeStatement(self, stmt_str: str) -> Optional[Any]:
        return self.execute_statement(stmt_str)

    def create_keyspace(self) -> None:
        self.execute_statement(RLKeySpace)

    def createKeySpace(self) -> None:
        self.create_keyspace()

    def gen_uuid(self) -> uuid.UUID:
        return uuid.uuid1(clock_seq=1234)

    def genUuid(self) -> uuid.UUID:
        return self.gen_uuid()


_default_instance: DBUtility = DBUtility.get_instance()

cluster: Optional[Any] = None
session: Optional[Any] = None
connectFuture: Optional[Any] = None


def isInitialized() -> bool:
    return _default_instance.is_initialized()


def is_initialized() -> bool:
    return _default_instance.is_initialized()


def initialize() -> None:
    global cluster, session, connectFuture
    _default_instance.initialize()
    cluster = _default_instance.cluster
    session = _default_instance.session
    connectFuture = _default_instance.connect_future


def terminate() -> None:
    global cluster, session, connectFuture
    _default_instance.terminate()
    cluster = None
    session = None
    connectFuture = None


def executeStatement(stmtStr: str) -> Optional[Any]:
    return _default_instance.execute_statement(stmtStr)


def execute_statement(stmt_str: str) -> Optional[Any]:
    return _default_instance.execute_statement(stmt_str)


def createKeySpace() -> None:
    _default_instance.create_keyspace()


def create_keyspace() -> None:
    _default_instance.create_keyspace()


def genUuid() -> uuid.UUID:
    return _default_instance.gen_uuid()


def gen_uuid() -> uuid.UUID:
    return _default_instance.gen_uuid()