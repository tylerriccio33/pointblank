from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Column:
    """
    A class to represent a column in a table.
    """

    name: str

    def __repr__(self):
        return self.name


def col(name: str) -> Column:
    """
    Reference a column in the input table.

    Parameters
    ----------
    name
        The name of the column.

    Returns
    -------
    Column
        A `Column` object representing the column.
    """
    return Column(name=name)
