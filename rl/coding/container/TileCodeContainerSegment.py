from typing import Optional, List, Union

from ... import db
from ...declares import SEGMENT_SIZE
from ...utility.CRUDInterface import CRUDInterface
from ...utility.IndexAccessorInterface import IndexAccessorInterface
from .TileCodeContainerCell import TileCodeContainerCell

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