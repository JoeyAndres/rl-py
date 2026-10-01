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
from typing import Any, Optional, Union

# Database connection placeholder or external module import
try:
    from rl import db
except ImportError:
    try:
        import db
    except ImportError:
        class _DbNamespace:
            session: Any = None
        db = _DbNamespace()

FLOAT = float


class TileCodeContainerCell:
    """
    Represents a single data that can be modified or retrieved.
    """

    def __init__(self, tileCodeContainerId: str, segmentIndex: int, index: int) -> None:
        """
        :param tileCodeContainerId: ID of the parent TileCodeContainer.
        :param segmentIndex: Index of this segment.
        :param index: Index of the data within segment.
        """
        self._tileCodeContainerId: str = tileCodeContainerId
        self._segmentIndex: int = segmentIndex
        self._index: int = index

    def get(self) -> float:
        """
        :return: the data being represented in this cell.
        """
        data_field = f"data{self._index}"
        stmt_str = (
            f"SELECT tileCodeContainerId, segmentIndex, {data_field}\n"
            f"FROM rl.tilecodecontainersegment\n"
            f"WHERE tileCodeContainerId = '{self._tileCodeContainerId}' AND\n"
            f"      segmentIndex = {self._segmentIndex};\n"
        )

        if db.session is None:
            sys.stderr.write("Query result: Database session is not initialized.\n")
            raise RuntimeError("Query failed.")

        try:
            result = db.session.execute(stmt_str)
        except Exception as e:
            sys.stderr.write(f"Query result: {e}\n")
            raise RuntimeError("Query failed.") from e

        if result is None:
            sys.stderr.write("Query result: NULL\n")
            raise RuntimeError("Query failed.")

        row = None
        if hasattr(result, "one"):
            row = result.one()
        elif hasattr(result, "current_rows") and result.current_rows:
            row = result.current_rows[0]
        else:
            try:
                row = next(iter(result))
            except (StopIteration, TypeError):
                row = None

        if row is None:
            return 0.0

        out_val = 0.0
        if hasattr(row, data_field):
            out_val = getattr(row, data_field)
        elif hasattr(row, data_field.lower()):
            out_val = getattr(row, data_field.lower())
        elif isinstance(row, dict):
            out_val = row.get(data_field, row.get(data_field.lower(), 0.0))
        elif isinstance(row, (tuple, list)):
            out_val = row[2] if len(row) > 2 else 0.0
        else:
            try:
                out_val = row[data_field]
            except Exception:
                out_val = 0.0

        return float(out_val) if out_val is not None else 0.0

    def set(self, val: float) -> None:
        """
        :param val: The value to set this cell.
        """
        data_field = f"data{self._index}"
        stmt_str = (
            f"UPDATE rl.tilecodecontainersegment\n"
            f"SET {data_field} = {val}\n"
            f"WHERE tileCodeContainerId='{self._tileCodeContainerId}' AND\n"
            f"      segmentIndex = {self._segmentIndex};\n"
        )

        if db.session is None:
            sys.stderr.write("Query result: Database session is not initialized.\n")
            raise RuntimeError("Query failed.")

        try:
            db.session.execute(stmt_str)
        except Exception as e:
            sys.stderr.write(f"Query result: {e}\n")
            raise RuntimeError("Query failed.") from e

    def __iadd__(self, val: Union[float, int, "TileCodeContainerCell"]) -> TileCodeContainerCell:
        self.set(self.get() + float(val))
        return self

    def __float__(self) -> float:
        return float(self.get())

    def __int__(self) -> int:
        return int(self.get())

    def __add__(self, other: Union[float, int, "TileCodeContainerCell"]) -> float:
        return self.get() + float(other)

    def __radd__(self, other: Union[float, int, "TileCodeContainerCell"]) -> float:
        return float(other) + self.get()

    def __sub__(self, other: Union[float, int, "TileCodeContainerCell"]) -> float:
        return self.get() - float(other)

    def __rsub__(self, other: Union[float, int, "TileCodeContainerCell"]) -> float:
        return float(other) - self.get()

    def __mul__(self, other: Union[float, int, "TileCodeContainerCell"]) -> float:
        return self.get() * float(other)

    def __rmul__(self, other: Union[float, int, "TileCodeContainerCell"]) -> float:
        return float(other) * self.get()

    def __truediv__(self, other: Union[float, int, "TileCodeContainerCell"]) -> float:
        return self.get() / float(other)

    def __rtruediv__(self, other: Union[float, int, "TileCodeContainerCell"]) -> float:
        return float(other) / self.get()

    def __neg__(self) -> float:
        return -self.get()

    def __pos__(self) -> float:
        return +self.get()

    def __abs__(self) -> float:
        return abs(self.get())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TileCodeContainerCell):
            return (
                self._tileCodeContainerId == other._tileCodeContainerId
                and self._segmentIndex == other._segmentIndex
                and self._index == other._index
            )
        try:
            return self.get() == float(other)  # type: ignore
        except (TypeError, ValueError):
            return False

    def __lt__(self, other: Union[float, int, "TileCodeContainerCell"]) -> bool:
        return self.get() < float(other)

    def __le__(self, other: Union[float, int, "TileCodeContainerCell"]) -> bool:
        return self.get() <= float(other)

    def __gt__(self, other: Union[float, int, "TileCodeContainerCell"]) -> bool:
        return self.get() > float(other)

    def __ge__(self, other: Union[float, int, "TileCodeContainerCell"]) -> bool:
        return self.get() >= float(other)

    def __repr__(self) -> str:
        return (
            f"TileCodeContainerCell("
            f"tileCodeContainerId={self._tileCodeContainerId!r}, "
            f"segmentIndex={self._segmentIndex!r}, "
            f"index={self._index!r})"
        )

    def __str__(self) -> str:
        return str(self.get())


spTileCodeContainerCell = Optional[TileCodeContainerCell]

__all__ = ["TileCodeContainerCell", "spTileCodeContainerCell", "FLOAT"]