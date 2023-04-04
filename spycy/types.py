from dataclasses import dataclass
from typing import Tuple


@dataclass
class Node:
    id_: int


@dataclass
class Edge:
    id_: Tuple[int, int, int]
