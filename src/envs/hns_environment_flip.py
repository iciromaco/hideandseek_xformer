"""
軽量ラッパー: `TeamCosEnv` の動作を変更せずに、外部から与える前進入力の符号を反転します。
元の `src/envs/hns_environment.py` は変更しません。

使い方:
    from src.envs.hns_environment_flip import FlippedForwardTeamCosEnv
    env = FlippedForwardTeamCosEnv(...)

このクラスは `step()` をオーバーライドして、渡された action の先頭要素 (forward)
だけを反転して親の `step()` に渡します。その他の振る舞いは完全に継承されます。
"""
from typing import Any
import numpy as np

from .hns_environment import TeamCosEnv


class FlippedForwardTeamCosEnv(TeamCosEnv):
    """TeamCosEnv を継承し、外部アクションの前進成分を反転するラッパー。

    注意: 内部で使われるモデル推論などは変更しません —
    外部から `env.step(action)` で与えられる `action[0]` の符号のみ反転します。
    """

    def step(self, action: Any):
        # NumPy の array/リスト/tuple 等を安全に扱い、元の action を破壊しない
        try:
            af = np.ravel(np.asarray(action, dtype=np.float32)).copy()
        except Exception:
            # 万一変換できない場合はそのまま親に渡す
            return super().step(action)

        if af.size >= 1:
            af[0] = -af[0]

        # 親の step に配列を渡す
        return super().step(af)
