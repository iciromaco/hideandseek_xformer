# Workspace Structure (main27中心)

## 1) 現行運用（main27）
- 実行エントリ: `main27_train_final.py`
- 実装本体: `src/`
- 設定: `configs/`
- 補助スクリプト: `scripts/`, `setenvX.sh`, `launch2.sh`

## 2) 旧バージョン/過去資産
- `legacy/versions/`
  - `main18_optimization.py`
  - `main25_HideAndSeek_HS.py`
  - `main25_HideAndSeek_HSR.py`
  - `main25_train_runner.py`
  - `main26_train_final.py`
  - `hide_and_seek_env.py`
  - `ppo_transformer.py`
  - `seeker_course.py`
- `legacy/archives/`
  - `main22以前/`
  - `main23/`

## 3) 引き継ぎ/説明資料
- `docs/handover/`
  - `HideAndSeek_Transformer プロジェクト引き継ぎ資料`
  - `開発・演習引き継ぎ資料.pdf`
  - `演習コンテンツ.txt`
  - `書き出されたCotEditor設定アーカイブ.cotsettings/`

## 4) 学習生成物・ログ（従来どおり）
- `checkpoints/`, `wandb/`, `runs/`, `logs/`
- 解析/可視化関連: `figures/`, `analysis_output_initial/`, `lidar_debug/`, `sightmap_debug/`, `sensitivity_results*/`

## 5) ルール（今後）
- 新規開発対象は `main27_train_final.py` と `src/`, `configs/` を優先。
- main27以外の新規試作は `legacy/` ではなく `scripts/` または `src/` 配下で管理。
- 旧版を再編集する場合は `legacy/` で完結させ、現行運用へは直接混在させない。
