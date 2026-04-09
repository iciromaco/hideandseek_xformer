# HideAndSeek v27 採用メモ（2026-03-06）

## 結論（採用）
- 現時点の標準学習条件は **wall_only 方針** を採用する。
- 具体的には、`RW_MOVE_SAT_PENALTY = 0.0`（移動飽和抑制なし）を前提に、他条件は公平比較時と同一（`TRAIN_OTHER_HIDER_POLICY = "rule"` など）で運用する。

## 比較条件
- 環境: `s1_h2_b2_r1`
- 学習対象: `hider`
- 公平条件: `TRAIN_OTHER_HIDER_POLICY = "rule"`, `TRAIN_OTHER_SEEKER_POLICY = "rule"`
- 長尺比較: 各条件 `100000 step × 3 seed`

## 100k×3 seed 比較結果（平均）

| 条件 | train/avg_reward | train/hide_rate | env/avg_blocked_ramp | env/lock_events | env/grab_events |
|---|---:|---:|---:|---:|---:|
| wall_only_long | **1.1820** | **0.5932** | 0.4674 | 35.0 | **3.7** |
| base_reward_long | 1.1109 | 0.5799 | 0.5745 | 61.0 | 20.0 |
| no_wall_stick_long | 0.9728 | 0.4901 | **0.4023** | 36.3 | 9.7 |

### 解釈
- 主目標（`avg_reward`, `hide_rate`）は `wall_only_long` が最良。
- 安定性指標（`lock/grab`）でも `wall_only_long` は良好。
- `no_wall_stick_long` は `avg_blocked_ramp` は低いが、主目標の成績が落ちる。

## 300k 確認ラン（採用条件）
- Run: `wall_only_confirm_300k`（W&B run id: `8qtmybzl`）
- 最終値:
  - `global_step = 299520`
  - `train/avg_reward = 0.9045`
  - `train/hide_rate = 0.5281`
  - `env/avg_blocked_ramp = 0.1000`
  - `env/lock_events = 49`
  - `env/grab_events = 23`

## 運用方針（当面）
1. `runtime.train` は wall_only 条件を維持。
2. 以後の変更は「100k×3 seed」で再比較してから本番採用。
3. 監視指標は `avg_reward`, `hide_rate`, `lock_events`, `grab_events` を必須とする。
