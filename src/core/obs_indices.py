# src/core/obs_indices.py
"""
Observation Vector Index Definitions for Hide and Seek.
LiDARの角度定義 [0, 15(L), -15(R), 30(L), -30(R), 45(L), -45(R), ...] に基づき、
物理的な意味を持つインデックスリストを定義します。
"""

from typing import Final, List


class SelfSchema:
    """自己情報用スキーマ (5次元固定)

    フィールド説明:
    - `VEL_X`, `VEL_Y`: エージェント基準（body-frame）で表現した重心速度成分。
      具体的にはワールド座標系の速度をエージェントの向きに対して
      -rot で回転して得た成分（`VEL_X` は前方成分、`VEL_Y` は左方成分）。
    - `ROT`: ワールド座標系のヨー角（ラジアン）。
    - `COS_ROT` / `SIN_ROT`: `cos(ROT)` / `sin(ROT)`。

    注: `VEL_*` は body-frame 表現、`ROT` は world-frame 表現という点で
    フレームが混在しているので取り扱いに注意すること。
    """

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
        self.IS_MOVING: Final = start + 6
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
        # total dimension of the observation vector
        self.total_dim = cursor

        # スクリプト型エージェント用の便宜上のグループ分け：シーカーとハイダーに対応する
        # AgentSchemaエントリのリスト。これらは、`set_others_keys(ens)`を呼び出すことで
        # 生成されます。ここで、`ens`は観測を構築する際に使用されるエージェントキーの
        # 順序付きリストです（`env._get_obs()`が構築する`ens`と同じものです）。

        self.SEEKERS: List[AgentSchema] = []
        self.HIDERS: List[AgentSchema] = []

    def set_others_keys(self, ens: List[str]):
        """ 指定された`ens` の順序に基づいて、`SEEKERS` および `HIDERS` リストを初期化します。

        `ens` は、呼び出し元が OTHERS エントリを生成する際に使用した
        エージェントキーのリストと同じものである必要があります（例えば、
        `TeamCosEnv._get_obs` 内で構築された `ens` など）。最初の `len(self.OTHERS)` 個のエントリのみが
        考慮されます。
        """
        # reset
        self.SEEKERS = []
        self.HIDERS = []
        for i, k in enumerate(ens[: len(self.OTHERS)]):
            if k.startswith("s"):
                self.SEEKERS.append(self.OTHERS[i])
            elif k.startswith("h"):
                self.HIDERS.append(self.OTHERS[i])

    def enemies_for_agent(self, agent_key: str):
        """指定された `agent_key` に対応する *敵* を、そのエージェントの視点から表す AgentSchema エントリのリストを返す。

        注：`SEEKERS` / `HIDERS` は `OTHERS` から派生しているため、
        観察しているエージェント自身は除外されます（`OTHERS` には
        自身は含まれないため）。観察エージェント自身の知覚を含める必要がある呼び出し元は、
        そのエージェントの観測情報を
        `visible_enemies_from_obs`（後述）に渡す必要があります。このヘルパー関数は
        渡された `obs` を確認するため、学習者自身の視点が考慮されます。
        """
        if agent_key is None:
            return []
        return self.HIDERS if agent_key.startswith("s") else self.SEEKERS

    def visible_enemies_from_obs(self, obs, agent_key: str):
        """`agent_key` に対する観測ベクトル `obs` が与えられた場合、その
        観測において「可視」とマークされている敵の AgentSchema エントリの
        リストを返す。

        これは、観測しているエージェントがどの
        敵を見ることができるかを知りたい場合に推奨されるヘルパー関数です。この関数は、
        敵チームの OTHERS エントリを適切に選択し、渡された `obs` 内の `VISIBLE` フラグで
        フィルタリングします（`obs` はそのエージェントの観測であるべきであるため、
        観測しているエージェント自身の視界も含まれます）。
        """
        enemies = self.enemies_for_agent(agent_key)
        if not enemies:
            return []
        return [en for en in enemies if float(obs[en.VISIBLE]) > 0.5]