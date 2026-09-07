"""Adapters shared by ArbiGraph task categories."""

from .adapter import Adapter
from .math_adapters import (
    AddScalarsAdapter,
    FullListDependencyAdapter,
    InterleaveListsAdapter,
    ListToListAdapter,
    ListToMatrixAdapter,
    ListToPolynomialAdapter,
    MatrixToListAdapter,
    PolynomialToListAdapter,
    ScalarToScalarAdapter,
    SumListsAdapter,
)

__all__ = [
    "Adapter",
    "AddScalarsAdapter",
    "FullListDependencyAdapter",
    "InterleaveListsAdapter",
    "ListToListAdapter",
    "ListToMatrixAdapter",
    "ListToPolynomialAdapter",
    "MatrixToListAdapter",
    "PolynomialToListAdapter",
    "ScalarToScalarAdapter",
    "SumListsAdapter",
]
