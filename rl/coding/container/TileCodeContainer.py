from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import (
    Any,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

# Type definitions matching declares.h
FLOAT = float
SEGMENT_SIZE: int = 100


# Base interface from rl.utility.CRUDInterface
class CRUDInterface(ABC):
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


# Database interface and fallback stubs matching rl::db
try:
    from rl import db  # type: ignore
except ImportError:
    class _DBStub:
        session: Any = None
        RLKeySpace: str = (
            "CREATE KEYSPACE IF NOT EXISTS rl\n"
            "  WITH replication = {'class': 'SimpleStrategy', 'replication_factor' : 3};"
        )
        TileCodeContainer: str = (
            "CREATE TABLE IF NOT EXISTS rl.tilecodecontainer (\n"
            "  id text PRIMARY KEY,\n"
            "  size bigint,\n"
            "  initialValue double\n"
            ");"
        )
        TileCodeContainerSegment: str = (
            "CREATE TABLE IF NOT EXISTS rl.tilecodecontainersegment (\n"
            "  tileCodeContainerId text,\n"
            "  size bigint,\n"
            "  segmentIndex bigint,\n"
            "  PRIMARY KEY (tileCodeContainerId, segmentIndex)"
            ") WITH caching = { 'keys' : 'NONE', 'rows_per_partition' : '120' };"
        )
        InsertTileCodeContainer: str = (
            "INSERT INTO rl.tilecodecontainer (id, size, initialValue) VALUES (?, ?, ?);"
        )

        @staticmethod
        def initialize() -> None:
            pass

        @staticmethod
        def terminate() -> None:
            pass

        @staticmethod
        def executeStatement(stmt: str) -> None:
            if _DBStub.session is not None:
                _DBStub.session.execute(stmt)

    db = _DBStub()


# Segment and Cell types matching rl::coding
try:
    from rl.coding.container.TileCodeContainerSegment import TileCodeContainerSegment  # type: ignore
except ImportError:
    class TileCodeContainerSegment(CRUDInterface):  # type: ignore
        def __init__(
            self,
            tile_code_container_id: str,
            size_or_index: int,
            index: Optional[int] = None,
        ) -> None:
            self.tile_code_container_id = tile_code_container_id
            if index is None:
                self.size = SEGMENT_SIZE
                self.index = size_or_index
            else:
                self.size = size_or_index
                self.index = index

        def create(self) -> None:
            pass

        def read(self) -> None:
            pass

        def update(self) -> None:
            pass

        def delete2(self) -> None:
            pass

        def at(self, i: int) -> Any:
            return TileCodeContainerCell()

        def __getitem__(self, i: int) -> Any:
            return self.at(i)


try:
    from rl.coding.container.TileCodeContainerCell import TileCodeContainerCell  # type: ignore
except ImportError:
    class TileCodeContainerCell:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._value: float = 0.0

        def set(self, value: float) -> None:
            self._value = value

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, (int, float)):
                return self._value == float(other)
            return super().__eq__(other)

        def __float__(self) -> float:
            return float(self._value)


T_ID = TypeVar("T_ID")
spTileCodeContainerSegment = TileCodeContainerSegment
TileCodeContainerAllocator = List[TileCodeContainerCell]


class TileCodeContainer(CRUDInterface, Generic[T_ID]):
    """Container type for TileCode weight vectors."""

    allocator_type = TileCodeContainerAllocator
    value_type = TileCodeContainerCell
    reference = TileCodeContainerCell
    const_reference = TileCodeContainerCell
    difference_type = int
    size_type = int

    _id: Optional[str] = None
    _size: int
    _initial_value: float = 0.0
    _segments: List[TileCodeContainerSegment]

    def __init__(
        self,
        size: Optional[Union[int, str]] = None,
        initial_value: float = 0.0,
        container_id: Optional[str] = None,
    ) -> None:
        """Creates an entry in db if not exist yet, otherwise retrieves the existing db."""
        if isinstance(size, str):
            self._id = size
            self._size = int(initial_value) if initial_value else 0
            self._initial_value = 0.0
        else:
            if container_id is not None:
                self._id = container_id
            self._size = size if size is not None else 0
            self._initial_value = float(initial_value)

        self._segments = []

        db.initialize()
        self.createSchema()

        if size is not None and not isinstance(size, str):
            if self.created():
                self.read()
            else:
                self.create()
        else:
            self.read()

    @classmethod
    def __class_getitem__(cls, item: Any) -> Any:
        if isinstance(item, tuple):
            id_str = "".join(str(c) for c in item)
        elif isinstance(item, str):
            id_str = item
        elif isinstance(item, TypeVar):
            return super().__class_getitem__(item)
        else:
            id_str = str(item)

        class _ParameterizedTileCodeContainer(cls):  # type: ignore
            _id = id_str

        _ParameterizedTileCodeContainer.__name__ = f"{cls.__name__}[{item!r}]"
        _ParameterizedTileCodeContainer.__qualname__ = f"{cls.__qualname__}[{item!r}]"
        return _ParameterizedTileCodeContainer

    def created(self) -> bool:
        """Returns True if already in db."""
        stmt_str = (
            "SELECT COUNT(*) as count\n"
            "FROM rl.tilecodecontainer\n"
            f"WHERE id='{self.getID()}';\n"
        )
        if db.session is not None:
            try:
                result = db.session.execute(stmt_str)
                row = result.one() if hasattr(result, "one") else (result[0] if result else None)
                if row is not None:
                    count = getattr(row, "count", row[0] if isinstance(row, (tuple, list)) else 0)
                    return bool(count)
            except Exception as e:
                raise RuntimeError("Query failed.") from e
        return False

    def create(self) -> None:
        if db.session is not None:
            try:
                db.session.execute(
                    db.InsertTileCodeContainer,
                    (self.getID(), self._size, float(self._initial_value)),
                )
            except Exception:
                try:
                    query = (
                        f"INSERT INTO rl.tilecodecontainer (id, size, initialValue) "
                        f"VALUES ('{self.getID()}', {self._size}, {float(self._initial_value)});"
                    )
                    db.session.execute(query)
                except Exception as e:
                    raise RuntimeError("Error creating TileCodeContainer") from e

        segment_count = self.getSegmentCount()
        for i in range(segment_count):
            _ = TileCodeContainerSegment(self.getID(), SEGMENT_SIZE, i)

        self._segments = self.getSegments()

    def read(self) -> None:
        stmt_str = (
            "SELECT size, initialValue\n"
            "FROM rl.tilecodecontainer\n"
            f"WHERE id='{self.getID()}';\n"
        )
        if db.session is not None:
            try:
                result = db.session.execute(stmt_str)
                row = result.one() if hasattr(result, "one") else (result[0] if result else None)
                if row is None:
                    raise RuntimeError("Query failed.")
                self._size = getattr(row, "size", row[0] if isinstance(row, (tuple, list)) else self._size)
                self._initial_value = getattr(
                    row,
                    "initialvalue",
                    getattr(
                        row,
                        "initialValue",
                        row[1] if isinstance(row, (tuple, list)) else self._initial_value,
                    ),
                )
            except Exception as e:
                raise RuntimeError("Query failed.") from e

        self._segments = self.getSegments()

    def update(self) -> None:
        pass

    def delete2(self) -> None:
        self._deleteAllSegments()

        stmt_str = (
            "DELETE FROM rl.tilecodecontainer\n"
            f"WHERE id='{self.getID()}';\n"
        )
        if db.session is not None:
            try:
                db.session.execute(stmt_str)
            except Exception as e:
                raise RuntimeError("Query failed.") from e

    def getID(self) -> str:
        """Returns container ID as string."""
        if self._id is not None:
            return str(self._id)
        return getattr(self, "ID", self.__class__.__name__)

    get_id = getID

    def getSegmentCount(self) -> int:
        """Returns segment count."""
        return math.ceil(float(self._size) / float(SEGMENT_SIZE))

    get_segment_count = getSegmentCount

    def getSegments(self) -> List[TileCodeContainerSegment]:
        """Returns array of TileCodeContainerSegment."""
        tile_code_container_segments: List[TileCodeContainerSegment] = []
        stmt_str = (
            "SELECT tileCodeContainerId, segmentIndex\n"
            "FROM rl.tilecodecontainersegment\n"
            f"WHERE tileCodeContainerId='{self.getID()}'\n"
            "ORDER BY segmentIndex ASC\n"
            "ALLOW FILTERING;\n"
        )
        if db.session is not None:
            try:
                result = db.session.execute(stmt_str)
                rows = result.all() if hasattr(result, "all") else list(result)
                for row in rows:
                    segment_index = getattr(
                        row,
                        "segmentindex",
                        getattr(
                            row,
                            "segmentIndex",
                            row[1] if isinstance(row, (tuple, list)) else 0,
                        ),
                    )
                    tile_code_container_segments.append(
                        TileCodeContainerSegment(self.getID(), segment_index)
                    )
            except Exception as e:
                raise RuntimeError("Query failed.") from e

        return tile_code_container_segments

    get_segments = getSegments

    def at(self, i: int) -> TileCodeContainerCell:
        """Returns TileCodeContainerCell entry in db."""
        segment_index = i // SEGMENT_SIZE
        inner_segment_index = i % SEGMENT_SIZE
        segment = self._segments[segment_index]
        if hasattr(segment, "at"):
            return segment.at(inner_segment_index)
        return segment[inner_segment_index]

    def __getitem__(self, i: int) -> TileCodeContainerCell:
        return self.at(i)

    def __setitem__(self, i: int, value: Any) -> None:
        cell = self.at(i)
        if hasattr(cell, "set"):
            cell.set(value)
        elif hasattr(cell, "value"):
            cell.value = value

    def __len__(self) -> int:
        return self._size

    def _deleteAllSegments(self) -> None:
        for segment in self.getSegments():
            segment.delete2()

    @classmethod
    def createSchema(cls) -> None:
        """Initialize the underlying database for TileCodeContainer."""
        cls._createTileContainerTable()

    @classmethod
    def destroySchema(cls) -> None:
        """Destroys the underlying database for TileCodeContainer."""
        pass

    @classmethod
    def _createTileContainerTable(cls) -> None:
        db.executeStatement(db.TileCodeContainer)

    @staticmethod
    def _insertTileCodeContainer(size: int, initial_value: float) -> str:
        raise NotImplementedError

    @staticmethod
    def _getTileContainer(uuid: str) -> Tuple[str, int, float]:
        raise NotImplementedError


spTileCodeContainer = TileCodeContainer