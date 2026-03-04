# src/core/obs_indices.py
"""
Observation Vector Index Definitions for Hide and Seek.
構成数（エージェント、オブジェクトの数）に応じて観測ベクトルのインデックスを
動的に事前計算し、obs[idx.B[0].REL_X] のような直感的なアクセスを提供します。
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
        self.SLICE: Final = slice(start, start + 7)
        self.REL_X: Final = start + 0
        self.REL_Y: Final = start + 1
        self.VEL_X: Final = start + 2
        self.VEL_Y: Final = start + 3
        self.QUAT: Final = start + 4
        self.IS_MOVING: Final = start + 5
        self.VISIBLE: Final = start + 6


class ObsIdx:
    """
    構成数に基づいて観測インデックスを動的に生成するマッパー。
    """
    def __init__(self, n_boxes: int, n_ramps: int, n_others: int):
        # 1. 自己情報 (0-4)
        self.SELF = SelfSchema(0)
        
        # 2. LiDAR (5-16)
        self.LIDAR = slice(5, 17)
        
        # 3. オブジェクト (17〜)
        cursor = 17
        self.B: List[ObjectSchema] = []
        for _ in range(n_boxes):
            self.B.append(ObjectSchema(cursor))
            cursor += 8
            
        self.RAMP: List[ObjectSchema] = []
        for _ in range(n_ramps):
            self.RAMP.append(ObjectSchema(cursor))
            cursor += 8
            
        # 4. 他エージェント
        self.OTHERS: List[AgentSchema] = []
        for _ in range(n_others):
            self.OTHERS.append(AgentSchema(cursor))
            cursor += 7
            
        # 最終的な観測次元
        self.total_dim = cursor

# --- 使用例 (環境やエージェントの __init__ 内でインスタンス化) ---
# idx = ObsIdx(n_boxes=2, n_ramps=1, n_others=2)
# obs_dim = idx.total_dim
# print(obs[idx.OTHERS[0].VISIBLE])