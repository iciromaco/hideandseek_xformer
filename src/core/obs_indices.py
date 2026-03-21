# src/core/obs_indices.py
"""
Observation Vector Index Definitions for Hide and Seek.
LiDARの角度定義 [0, 15(L), -15(R), 30(L), -30(R), 45(L), -45(R), ...] に基づき、
物理的な意味を持つインデックスリストを定義します。
"""

from typing import Final, List


class SelfSchema:
    """自己情報用スキーマ (5次元固定)"""
    def __init__(self, start: int = 0):
        self.SLICE: Final = slice(start, start + 5)
        self.VEL_X: Final = start + 0
        self.VEL_Y: Final = start + 1
        self.ROT: Final = start + 2
        self.COS_ROT: Final = start + 3
        self.SIN_ROT: Final = start + 4


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
    """他エージェント用スキーマ (7次元)"""
    def __init__(self, start: int):
        self.SLICE: Final = slice(start, start + 8)
        self.REL_X: Final = start + 0
        self.REL_Y: Final = start + 1
        self.VEL_X: Final = start + 2
        self.VEL_Y: Final = start + 3
        self.QUAT_0: Final = start + 4  # cos(rot)
        self.QUAT_1: Final = start + 5  # sin(rot)
        self.BEING_HIT: Final = start + 6
        self.VISIBLE: Final = start + 7


class ObsIdx:
    """
    構成数に基づいて観測インデックスを動的に生成するマッパー。
    LiDARの方向定義をここに集約し、思い込みによるミスを防止します。
    """
    def __init__(self, n_boxes: int, n_ramps: int, n_others: int):
        self.SELF = SelfSchema(0)
        self.LIDAR = slice(5, 17)

        # LiDAR 内部インデックス (VisibilityEngineのanglesに対応)
        # angles = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
        self.LIDAR_FRONT_IDX: Final = [0, 1, 2]         # 0°, 15°, -15°
        self.LIDAR_LEFT_IDX: Final = [1, 3, 5, 7, 9]    # 15°, 30°, 45°, 90°, 135°
        self.LIDAR_RIGHT_IDX: Final = [2, 4, 6, 8, 10]  # -15°, -30°, -45°, -90°, -135°
        self.LIDAR_BACK_IDX: Final = [11]               # 180°

        cursor = 17
        self.B: List[ObjectSchema] = []
        for _ in range(n_boxes):
            self.B.append(ObjectSchema(cursor))
            cursor += 8

        self.RAMP: List[ObjectSchema] = []
        for _ in range(n_ramps):
            self.RAMP.append(ObjectSchema(cursor))
            cursor += 8

        self.OTHERS: List[AgentSchema] = []
        for _ in range(n_others):
            self.OTHERS.append(AgentSchema(cursor))
            cursor += 8

        self.total_dim = cursor