"""
core obs_indices.py
Observation Vector Index Definitions for Hide and Seek.
インスタンスを直接エクスポートすることで、obs[SELF.VEL_X] のような
最短の記述を実現します。
"""

from typing import Final

class SelfSchema:
    """自己情報用スキーマ (0-4次元)"""
    def __init__(self):
        self.SLICE: Final = slice(0, 5)
        self.VEL_X: Final = 0
        self.VEL_Y: Final = 1
        self.ROT: Final = 2
        self.COS_ROT: Final = 3
        self.SIN_ROT: Final = 4


class ObjectSchema:
    """箱やスロープなどのオブジェクト用スキーマ (8次元)"""
    def __init__(self, start: int):
        self.SLICE: Final = slice(start, start + 8)
        self.REL_X: Final = start + 0
        self.REL_Y: Final = start + 1
        self.VEL_X: Final = start + 2
        self.VEL_Y: Final = start + 3
        self.QUAT_0: Final = start + 4
        self.QUAT_1: Final = start + 5
        self.IS_MOVING: Final = start + 6
        self.IS_LOCKED: Final = start + 7


class AgentSchema:
    """敵エージェント用スキーマ (7次元)"""
    def __init__(self, start: int):
        self.SLICE: Final = slice(start, start + 7)
        self.REL_X: Final = start + 0
        self.REL_Y: Final = start + 1
        self.VEL_X: Final = start + 2
        self.VEL_Y: Final = start + 3
        self.QUAT: Final = start + 4
        self.IS_MOVING: Final = start + 5
        self.VISIBLE: Final = start + 6


# --- インスタンスを直接定義（ここがポイント） ---
SELF: Final = SelfSchema()
LIDAR: Final = slice(5, 17)

B1: Final = ObjectSchema(17)
B2: Final = ObjectSchema(25)
RAMP: Final = ObjectSchema(33)

H1: Final = AgentSchema(41)
H2: Final = AgentSchema(48)