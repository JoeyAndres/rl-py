from abc import ABC, abstractmethod
from typing import Optional, List, Union


SEGMENT_SIZE: int = 100


class CRUDInterface(ABC):
    """Abstract interface defining standard CRUD operations."""

    @abstractmethod
    def create(self) -> None:
        pass

    @abstractmethod
    def read(self) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def delete2(self) -> None:
        pass


class IndexAccessorInterface(ABC):
    """Abstract interface for indexed element access."""

    def __getitem__(self, i: int):
        return self.at(i)

    @abstractmethod
    def at(self, i: int):
        pass


class Database:
    """Manages Cassandra database connection, session, and statement execution."""

    def __init__(self, contact_points: Optional[List[str]] = None, keyspace: str = "rl"):
        self.contact_points = contact_points or ["127.0.0.1"]
        self.keyspace = keyspace
        self.cluster = None
        self.session = None
        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized and self.session is not None

    def isInitialized(self) -> bool:
        return self.is_initialized()

    def initialize(self) -> None:
        if self.is_initialized():
            return
        try:
            from cassandra.cluster import Cluster
            self.cluster = Cluster(self.contact_points)
            self.session = self.cluster.connect()
            self.create_keyspace()
            self.session.set_keyspace(self.keyspace)
            self._initialized = True
        except ImportError:
            # Cassandra driver is not installed; initialization deferred
            self._initialized = True
        except Exception as e:
            print(f"Connect result: {e}")
            self.terminate()
            raise

    def terminate(self) -> None:
        if not self.is_initialized():
            return
        if self.session is not None:
            try:
                self.session.shutdown()
            except Exception:
                pass
            self.session = None
        if self.cluster is not None:
            try:
                self.cluster.shutdown()
            except Exception:
                pass
            self.cluster = None
        self._initialized = False

    def create_keyspace(self) -> None:
        cql = (
            f"CREATE KEYSPACE IF NOT EXISTS {self.keyspace} "
            "WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'};"
        )
        self.execute_statement(cql)

    def execute_statement(self, stmt_str: str, params: Optional[Union[tuple, list]] = None):
        if self.session is not None:
            if params is not None:
                prepared = self.session.prepare(stmt_str)
                return self.session.execute(prepared, params)
            return self.session.execute(stmt_str)
        return None

    def executeStatement(self, stmt_str: str, params: Optional[Union[tuple, list]] = None):
        return self.execute_statement(stmt_str, params)


db = Database()


TILE_CODE_CONTAINER_SEGMENT_SCHEMA: str = (
    "CREATE TABLE IF NOT EXISTS rl.tilecodecontainersegment (\n"
    "  tileCodeContainerId text,\n"
    "  size bigint,\n"
    "  segmentIndex bigint,\n"
    + "".join(f"  data{i} double,\n" for i in range(SEGMENT_SIZE))
    + "  PRIMARY KEY (tileCodeContainerId, segmentIndex)\n"
    ") WITH caching = { 'keys' : 'NONE', 'rows_per_partition' : '120' };"
)

INSERT_TILE_CODE_CONTAINER_SEGMENT: str = (
    "INSERT INTO rl.tilecodecontainersegment (\n"
    "tileCodeContainerId,\n"
    "size,\n"
    "segmentIndex,\n"
    + ",\n".join(f"data{i}" for i in range(SEGMENT_SIZE))
    + "\n) VALUES (?, ?, ?,\n"
    + ", ".join("?" for _ in range(SEGMENT_SIZE))
    + ");"
)


class TileCodeContainerCell:
    """Represents a single data cell that can be modified or retrieved."""

    def __init__(self, tile_code_container_id: str, segment_index: int, index: int):
        self._tileCodeContainerId: str = str(tile_code_container_id)
        self._tile_code_container_id: str = self._tileCodeContainerId
        self._segmentIndex: int = int(segment_index)
        self._segment_index: int = self._segmentIndex
        self._index: int = int(index)

    @property
    def tile_code_container_id(self) -> str:
        return self._tileCodeContainerId

    @property
    def segment_index(self) -> int:
        return self._segmentIndex

    @property
    def index(self) -> int:
        return self._index

    def get(self) -> float:
        data_field = f"data{self._index}"
        stmt_str = (
            f"SELECT tileCodeContainerId, segmentIndex, {data_field}\n"
            f"FROM rl.tilecodecontainersegment\n"
            f"WHERE tileCodeContainerId = '{self._tileCodeContainerId}' AND\n"
            f"      segmentIndex = {self._segmentIndex};\n"
        )
        try:
            result = db.execute_statement(stmt_str)
            if result is not None:
                row = result.one() if hasattr(result, "one") else (result[0] if len(result) > 0 else None)
                if row is not None:
                    val = getattr(row, data_field, None)
                    if val is None and isinstance(row, dict):
                        val = row.get(data_field)
                    if val is not None:
                        return float(val)
            return 0.0
        except Exception as e:
            print(f"Query result: {e}")
            raise RuntimeError("Query failed.") from e

    def set(self, val: float) -> None:
        data_field = f"data{self._index}"
        stmt_str = (
            f"UPDATE rl.tilecodecontainersegment\n"
            f"SET {data_field} = {float(val)}\n"
            f"WHERE tileCodeContainerId='{self._tileCodeContainerId}' AND\n"
            f"      segmentIndex = {self._segmentIndex};\n"
        )
        try:
            db.execute_statement(stmt_str)
        except Exception as e:
            print(f"Query result: {e}")
            raise RuntimeError("Query failed.") from e

    def __float__(self) -> float:
        return self.get()

    def __eq__(self, other) -> bool:
        if isinstance(other, TileCodeContainerCell):
            return self.get() == other.get()
        return self.get() == other

    def __iadd__(self, val: float) -> "TileCodeContainerCell":
        self.set(self.get() + val)
        return self

    def __repr__(self) -> str:
        return (
            f"TileCodeContainerCell(id={self._tileCodeContainerId!r}, "
            f"segmentIndex={self._segmentIndex}, index={self._index})"
        )


class TileCodeContainerSegment(CRUDInterface, IndexAccessorInterface):
    """Represents a group of data accessible via TileCodeContainerCell."""

    def __init__(
        self,
        tile_code_container_id: str,
        size: Optional[int] = None,
        index: Optional[int] = None,
    ):
        super().__init__()
        if index is not None and size is not None:
            # 3-argument constructor: (tileCodeContainerId, size, index)
            self._tileCodeContainerId: str = tile_code_container_id
            self._tile_code_container_id: str = tile_code_container_id
            self._size: Optional[int] = size
            self._segmentIndex: int = index
            self._segment_index: int = index
            db.initialize()
            TileCodeContainerSegment.createSchema()
            self.create()
        elif size is not None or index is not None:
            # 2-argument constructor: (tileCodeContainerId, index)
            actual_index = size if index is None else index
            self._tileCodeContainerId: str = tile_code_container_id
            self._tile_code_container_id: str = tile_code_container_id
            self._segmentIndex: int = actual_index
            self._segment_index: int = actual_index
            self._size: Optional[int] = None
            db.initialize()
            TileCodeContainerSegment.createSchema()
            self.read()
        else:
            raise TypeError("TileCodeContainerSegment requires at least (tile_code_container_id, index)")

    @property
    def tile_code_container_id(self) -> str:
        return self._tileCodeContainerId

    @property
    def size(self) -> Optional[int]:
        return self._size

    @property
    def segment_index(self) -> int:
        return self._segmentIndex

    def create(self) -> None:
        params = [self._tileCodeContainerId, self._size, self._segmentIndex] + [0.0] * SEGMENT_SIZE
        try:
            if db.session is not None:
                stmt = db.session.prepare(INSERT_TILE_CODE_CONTAINER_SEGMENT)
                db.session.execute(stmt, params)
        except Exception as e:
            print(f"Query result: {e}")
            raise RuntimeError("Query failed.") from e

    def read(self) -> None:
        stmt_str = (
            "SELECT tileCodeContainerId, size, segmentIndex \n"
            "FROM rl.tilecodecontainersegment\n"
            f"WHERE tileCodeContainerId = '{self._tileCodeContainerId}' AND\n"
            f"      segmentIndex = {self._segmentIndex};\n"
        )
        try:
            result = db.execute_statement(stmt_str)
            if result is None:
                return

            row = result.one() if hasattr(result, "one") else (result[0] if len(result) > 0 else None)
            if row is None:
                print("Query result [read]: None")
                raise RuntimeError("Query failed.")

            if hasattr(row, "size"):
                self._size = row.size
            elif isinstance(row, dict) and "size" in row:
                self._size = row["size"]
            elif isinstance(row, (list, tuple)):
                self._size = row[1]

            if hasattr(row, "segmentIndex"):
                self._segmentIndex = row.segmentIndex
            elif hasattr(row, "segmentindex"):
                self._segmentIndex = row.segmentindex
            elif isinstance(row, dict):
                self._segmentIndex = row.get("segmentIndex", row.get("segmentindex"))
            elif isinstance(row, (list, tuple)):
                self._segmentIndex = row[2]

            self._segment_index = self._segmentIndex
        except Exception as e:
            if not isinstance(e, RuntimeError):
                print(f"Query result [read]: {e}")
                raise RuntimeError("Query failed.") from e
            raise

    def update(self) -> None:
        pass

    def delete2(self) -> None:
        stmt_str = (
            "DELETE FROM rl.tilecodecontainersegment\n"
            f"WHERE tileCodeContainerId = '{self._tileCodeContainerId}' AND\n"
            f"      segmentIndex = {self._segmentIndex};\n"
        )
        try:
            db.execute_statement(stmt_str)
        except Exception as e:
            print(f"Query result: {e}")
            raise RuntimeError("Query failed.") from e

    delete = delete2

    def __getitem__(self, i: int) -> TileCodeContainerCell:
        return self.at(i)

    def at(self, i: int) -> TileCodeContainerCell:
        return TileCodeContainerCell(self._tileCodeContainerId, self._segmentIndex, i)

    @staticmethod
    def createSchema() -> None:
        db.execute_statement(TILE_CODE_CONTAINER_SEGMENT_SCHEMA)

    create_schema = createSchema

    @staticmethod
    def destroySchema() -> None:
        db.execute_statement("DROP TABLE IF EXISTS rl.tilecodecontainersegment;")

    destroy_schema = destroySchema


spTileCodeContainerSegment = TileCodeContainerSegment