import os
import sys
import json
import time
import subprocess
import re
from pathlib import Path
import platform
import shutil
import numpy as np

# --- 必要なライブラリのチェック ---
try:
    import psutil
except ImportError as e:
    print(f"Error: 必要なライブラリが見つかりません: {e}")
    print("以下のコマンドでインストールしてください:")
    print("uv add psutil numpy")
    sys.exit(1)

# ==========================================
# 1. 分析設定
# ==========================================

# 対象となるスクリプト名
# ※ ユーザー環境に合わせて最適化を行う対象
TARGET_SCRIPT = "main18_optimization.py"

# ベースラインパラメータ（基準となる設定）
# これを基準に、1つずつパラメータを変更して影響を計測します
BASE_PARAMS = {
    "NUM_ENVS": 8,              # 並列数
    "NUM_STEPS": 128,            # 1更新あたりのステップ数
    "TRANSFORMER_SEQ_LEN": 8,   # 記憶の長さ
    "HIDDEN_DIM": 64,           # ネットワークの大きさ
    "NUM_LAYERS": 2,            # Transformer層数
    "NUM_HEADS": 1,             # Attentionヘッド数
    "UPDATE_EPOCHS": 2,         # 1データあたりの学習回数
    "TOTAL_TIMESTEPS": 50000,   # 実験用の短いステップ数
}

# 感度分析を行うパラメータと候補値
PARAM_SENSITIVITY = {
    "NUM_ENVS": [4, 8],          # 並列化の効果を確認
    "NUM_STEPS": [64, 128, 256],       # データの鮮度と安定性のトレードオフ
    "HIDDEN_DIM": [32, 64, 128],       # モデルサイズの影響
    "TRANSFORMER_SEQ_LEN": [4, 8, 16], # 記憶長の影響
    "NUM_LAYERS": [1, 2],              # 深さの影響
    "NUM_HEADS": [1,2,4],             # Attentionヘッド数
    "UPDATE_EPOCHS": [2, 4],           # 学習強度の影響
}

# ★複数シード実験
# 偶然性を排除し、結果の安定性(ロバストネス)を評価するために
# 複数の乱数シードで実験を行います。
SEEDS = [1, 42, 1023, 2024, 9999]

# ==========================================
# 2. ユーティリティ関数
# ==========================================

def estimate_memory(num_envs, num_steps, hidden_dim, seq_len=16, obs_dim=53):
    """メモリ使用量を概算（MB単位）"""
    # 観測バッファ (double buffered)
    obs_buffer = num_steps * num_envs * seq_len * 2 * obs_dim * 4 / (1024**2)
    # その他のバッファ (actions, logprobs, rewards, values, dones)
    other_buffers = num_steps * num_envs * (4 + 1 + 1 + 1 + 1) * 4 / (1024**2)
    # モデルサイズ (概算)
    model_size = (hidden_dim * obs_dim + hidden_dim**2 * 4) * 4 / (1024**2)
    # PyTorch/CUDA overhead & Libraries & MuJoCo overhead
    overhead = 400 + (num_envs * 40) 
    
    total = (obs_buffer + other_buffers + model_size) * 3 + overhead
    return total

def create_test_script(params, output_dir, seed):
    """パラメータを書き換えた一時スクリプトを作成"""
    if not os.path.exists(TARGET_SCRIPT):
        raise FileNotFoundError(f"{TARGET_SCRIPT} が見つかりません。同じフォルダに配置してください。")

    with open(TARGET_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()

    # 正規表現を用いてパラメータを置換
    for key, val in params.items():
        # パターン: 行頭(空白許容) + 変数名 + 空白 + = + 空白 + 値 + コメント等
        pattern = fr"^(\s*{key}\s*=\s*)(.+?)(?=\s*(#.*)?$)"
        
        if not re.search(pattern, content, re.MULTILINE):
            print(f"  [Warning] 変数定義が見つかりません: {key}")
            continue
            
        # \g<1> を使用して置換 (数字との結合回避)
        replacement = fr"\g<1>{val}"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # 実験設定の強制上書き (公平な比較のため)
    # 1. 実験名を変更
    content = re.sub(r'EXPERIMENT_NAME = ".*?"', f'EXPERIMENT_NAME = "SensAnalysis_Seed{seed}"', content)
    # 2. WandB無効化
    content = re.sub(r'TRACK_WANDB = True', 'TRACK_WANDB = False', content)
    # 3. Viewer無効化
    content = re.sub(r'USE_VIEWER = True', 'USE_VIEWER = False', content)
    
    # ★重要: 比較実験のため、既存モデルのロードは無効化し、シードを固定する
    content = re.sub(r'LOAD_EXISTING_MODELS = True', 'LOAD_EXISTING_MODELS = False', content)
    content = re.sub(r'FIXED_SEED = None', f'FIXED_SEED = {seed}', content)
    # 万が一古い記述(SEED = int...)が残っていた場合の保険
    content = re.sub(r'SEED = int\(time\.time\(\)\)', f'SEED = {seed}', content)

    script_path = output_dir / "temp_run.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)

    return script_path

def run_training(params, trial_dir, seed, timeout=1800):
    """訓練を実行し、結果（時間と報酬）を返す"""
    script_path = None
    try:
        script_path = create_test_script(params, trial_dir, seed)

        # メモリチェック
        est_mem = estimate_memory(params["NUM_ENVS"], params["NUM_STEPS"], params["HIDDEN_DIM"], params["TRANSFORMER_SEQ_LEN"])
        avail_mem = psutil.virtual_memory().available / (1024**2)
        if est_mem > avail_mem * 0.85:
            print(f"    [Warning] High memory usage predicted ({est_mem:.0f}MB). Available: {avail_mem:.0f}MB")

        start_time = time.time()

        # サブプロセス実行
        # cwdは変更せず、ルートディレクトリで実行 (ImportError回避)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        elapsed_time = time.time() - start_time

        if result.returncode != 0:
            err_lines = result.stderr.strip().split('\n')[-3:]
            return {"success": False, "error": "\n".join(err_lines), "elapsed_time": elapsed_time}

        # 報酬のパース (正規表現で堅牢化)
        final_reward = None
        reward_pattern = r'Reward:\s*([-+]?\d*\.?\d+)'
        
        # ログの最後の数行から探す（最新の報酬）
        stdout_lines = result.stdout.split("\n")
        for line in reversed(stdout_lines):
            match = re.search(reward_pattern, line)
            if match:
                final_reward = float(match.group(1))
                break
        
        if final_reward is None:
             return {"success": False, "error": "Could not parse reward from output", "elapsed_time": elapsed_time}

        # 平均SPSの計算
        sps = params["TOTAL_TIMESTEPS"] / elapsed_time if elapsed_time > 0 else 0

        return {
            "success": True,
            "elapsed_time": elapsed_time,
            "final_reward": final_reward,
            "sps": sps,
            "error": None
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timeout (>{timeout}s)", "elapsed_time": timeout}
    except Exception as e:
        return {"success": False, "error": str(e), "elapsed_time": 0}
    finally:
        # 一時ファイルの削除
        if script_path and os.path.exists(script_path):
            try: os.remove(script_path)
            except: pass

def run_sensitivity_analysis_for_seed(seed, output_base):
    """指定されたシードで感度分析を実行"""
    print(f"\n{'='*70}")
    print(f"Seed = {seed}")
    print(f"{'='*70}")
    
    all_results = {}
    
    for param_name, candidates in PARAM_SENSITIVITY.items():
        print(f"\n[{param_name}] Testing candidates: {candidates}")
        param_results = []
        
        for value in candidates:
            # パラメータセットの作成
            current_params = BASE_PARAMS.copy()
            current_params[param_name] = value

            # ディレクトリ作成
            trial_name = f"{param_name}_{value}"
            trial_dir = output_base / trial_name
            trial_dir.mkdir(exist_ok=True)

            print(f"  > Trial: {value} ... ", end="", flush=True)
            
            # 実行
            res = run_training(current_params, trial_dir, seed)

            # 結果表示
            if res["success"]:
                print(f"Success! (SPS: {res['sps']:.1f}, Rew: {res['final_reward']:.3f})")
            else:
                print(f"Failed. ({res.get('error', 'Unknown')})")

            # データ保存用
            record = {
                "param_name": param_name,
                "param_value": value,
                "seed": seed,
                **res
            }
            param_results.append(record)

        all_results[param_name] = param_results
    
    return all_results

def generate_statistical_summary(all_seeds_results, output_base):
    """複数シードの結果から統計サマリーを生成"""
    
    summary_path = output_base / "statistical_summary.txt"
    
    with open(summary_path, "w", encoding="utf-8") as f:
        def log(text):
            print(text)
            f.write(text + "\n")
        
        log("\n" + "="*70)
        log(" STATISTICAL SUMMARY (MULTI-SEED ANALYSIS)")
        log("="*70)
        
        recommended_params = {}

        for param_name in PARAM_SENSITIVITY.keys():
            log(f"\n[ {param_name} ]")
            
            # 各パラメータ値ごとに報酬を集計
            param_stats = {}
            sps_stats = {}
            
            for seed, results in all_seeds_results.items():
                if param_name not in results: continue
                for record in results[param_name]:
                    value = record['param_value']
                    
                    if value not in param_stats:
                        param_stats[value] = []
                        sps_stats[value] = []
                    
                    if record.get('success'):
                        param_stats[value].append(record['final_reward'])
                        sps_stats[value].append(record['sps'])
            
            # 統計量を計算・表示
            log(f"  {'Value':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'SPS_Mean':<10}")
            log(f"  {'-'*58}")
            
            robustness_scores = {}
            
            for value in sorted(param_stats.keys()):
                rewards = param_stats[value]
                sps_vals = sps_stats[value]
                
                if rewards:
                    mean = np.mean(rewards)
                    std = np.std(rewards)
                    min_r = np.min(rewards)
                    max_r = np.max(rewards)
                    sps_mean = np.mean(sps_vals)
                    
                    log(f"  {str(value):<8} {mean:<10.3f} {std:<10.3f} {min_r:<10.3f} {max_r:<10.3f} {sps_mean:<10.1f}")
                    
                    # ロバストネス評価
                    # Mean / Std (シャープレシオ的な指標)
                    # 平均が高く、かつバラつき(Std)が小さいものを高く評価
                    robustness = mean / (std + 1e-6)
                    
                    robustness_scores[value] = {
                        "mean": mean,
                        "std": std,
                        "robustness": robustness
                    }
            
            if robustness_scores:
                # 最も堅牢なパラメータを選ぶ
                best_param = max(robustness_scores.keys(), 
                                key=lambda v: robustness_scores[v]["robustness"])
                
                stats = robustness_scores[best_param]
                log(f"\n  >> Recommended: {best_param} (Mean/Std: {stats['robustness']:.2f})")
                recommended_params[param_name] = best_param
        
        # 推奨パラメータセットを JSON で保存
        json_recommended_path = output_base / "recommended_params.json"
        with open(json_recommended_path, "w") as jf:
            json.dump(recommended_params, jf, indent=2, default=str)
        
        log(f"\nRecommended parameters saved to: recommended_params.json")

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    print("="*70)
    print("Sensitivity Analysis for HideAndSeek (Multi-Seed)")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Seeds: {SEEDS}")
    print("="*70)
    
    if not os.path.exists(TARGET_SCRIPT):
        print(f"Error: {TARGET_SCRIPT} not found.")
        print(f"Please ensure '{TARGET_SCRIPT}' is in the same directory.")
        return

    output_base = Path("sensitivity_results_multiseed")
    output_base.mkdir(exist_ok=True)
    
    # 前回の結果があればバックアップ
    json_path = output_base / "all_results.json"
    if json_path.exists():
        shutil.copy(json_path, output_base / "all_results_backup.json")

    # ★各シードで感度分析を実行
    all_seeds_results = {}
    
    for seed in SEEDS:
        seed_output_dir = output_base / f"seed_{seed}"
        seed_output_dir.mkdir(exist_ok=True)
        
        # シードごとの感度分析を実行
        results = run_sensitivity_analysis_for_seed(seed, seed_output_dir)
        all_seeds_results[seed] = results
        
        # 各シードの結果を JSON で保存
        seed_json_path = seed_output_dir / "results.json"
        with open(seed_json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
    
    # 全データ保存
    with open(json_path, "w") as f:
        json.dump(all_seeds_results, f, indent=2, default=str)

    # ★統計サマリーを生成
    generate_statistical_summary(all_seeds_results, output_base)
    
    print(f"\n{'='*70}")
    print("Analysis Complete.")
    print(f"Results saved to: {output_base}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()