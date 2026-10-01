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
from ...utility.CRUDInterface import CRUDInterface

# Type definitions matching declares.h
FLOAT = float
SEGMENT_SIZE: int = 100


# Database interface and fallback stubs matching rl::db
from ... import db


# Segment and Cell types matching rl::coding
from .TileCodeContainerSegment import TileCodeContainerSegment


from .TileCodeContainerCell import TileCodeContainerCell


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