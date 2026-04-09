環境⇄エージェント分離 API 仕様

目的
- 環境はポリシーの実装・ロード・履歴管理に依存しない。環境は物理シミュレーションとNPCの同期的実行、および観測の正規化とアクション適用に専念する。
- エージェント（学習モデルや外部ポリシー）は観測を受け取りアクションを返す外部コンポーネントとする。

基本方針
1. 観測とアクション
   - 観測フォーマット: 既存の `env._normalize_obs(env._get_obs(idx))` による1次元 float32 配列を標準とする。
   - アクションフォーマット: 既存の4要素配列 `[forward, turn, lock, grab]` を使用する。
   - 観測供給タイミング: `env.reset()` は学習対象の観測を返す。`env.step(action)` は与えられたアクション（学習対象）を適用し、(obs, reward, terminated, truncated, info) を返す。 

2. NPC（非学習エージェント）
   - NPC の意思決定ロジック（`RuleBasedSeeker`, `RuleBasedHider`）は環境内部に残す。
   - ただし意思決定の実装は `src/agents/scripted_agents.py` のまま外部モジュール化しておき、環境はそのモジュールの関数/クラスを呼ぶだけにする。
   - NPC は環境内部の物理状態（位置、速度、オブジェクト状態）に直接アクセスしてアクションを決定する。

3. ポリシー（学習モデル）
   - 環境はモデルのロード、重み適用、履歴管理を行わない。
   - 既存の `set_shared_team_policy_state`, `set_inference_policy_state` は廃止（あるいは互換性のため警告を出す shim にする）。
   - 外部の `PolicyManager`（例: `src/policy/policy_manager.py`）が、必要なら観測の履歴管理やバッチ化を行い、各 `env.step()` 呼び出しで `action` を渡す責務を持つ。

4. 履歴と時系列入力（Transformer 等）
   - モデルが過去観測を必要とする場合、履歴管理は環境側ではなく `PolicyManager` 側で行う。
   - ただし環境は `normalize_obs` を提供し、履歴を作るための始点となる観測を返す。初期化（reset）時に履歴プリムは `PolicyManager` が行う。

5. 互換性と移行戦略
   - フェーズ1（移行期）: `TeamCosEnv` に軽量互換 shim を残し、旧API呼び出しに対して警告と False を返す実装にする。外部 `PolicyManager` を導入して既存呼び出しを順次置換。
   - フェーズ2（完全分離）: shim 削除、全呼び出し箇所を外部 `PolicyManager` ベースに更新。

6. API メソッド一覧（最終目標）
   - `env.reset(seed=None, options=None) -> obs, info`
   - `env.step(action) -> obs, reward, terminated, truncated, info`
   - `env.render()` / `env.close()`（従来通り）
   - NPC関連は `env` の内部実装だが、プラグイン的に差し替え可能にする（コンストラクタ引数 `npc_factory=None` でクラス/ファクトリを注入できる）。

7. テスト/検証
   - smoke test: `scripts/smoke_env.py` を更新し、学習モデルを環境にロードする代わりに `PolicyManager` 経由で観測を取得し action を env.step に渡すシナリオを確認。
   - vector test: `scripts/vector_shared_policy_test.py` を `PolicyManager` ベースに書き換え、マルチプロセスでの外部ポリシー適用を確認。

実行計画（短期）
1. `src/envs/ENV_AGENT_API.md` を作成（本ファイル）。
2. `TeamCosEnv` を段階的にリファクタ（shim 残しつつ policy 管理削除）。
3. `src/policy/policy_manager.py` を導入・サンプル実装。
4. スクリプト（`scripts/*.py`）を順次更新して `PolicyManager` を使うように変更し、テスト実行。

次のステップ
- 進めてよければ次に `TeamCosEnv` からポリシー関連コードを削除するドラフトパッチを作成します。