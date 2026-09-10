from typing import NamedTuple


class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class IVec2(NamedTuple):
  x: int
  y: int