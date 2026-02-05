# inference.py
import os
from hideandseek import HideAndSeekEnv
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import time

# 推論対象を指定（"HIDER" または "SEEKER"）
os.environ["TRAIN_TARGET"] = "HIDER"  # ← ここで切り替え！

# モデルパス
MODEL_PATH = "xtransformer_hider_model.zip" if os.environ["TRAIN_TARGET"] == "HIDER" else "xtransformer_seeker_model.zip"

# 環境初期化（render_mode="human" で表示）
env = HideAndSeekEnv(train_target=os.environ["TRAIN_TARGET"], render_mode="human")

# モデル読み込み
model = RecurrentPPO.load(MODEL_PATH)

for _ in range(100):
    obs, _ = env.reset()
    state = None
    done = False
    while not done:
        loop_start = time.time()
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        step_duration = 0.033  # 例: 0.08秒くらい
        process_time = time.time() - loop_start
        wait_time = step_duration - process_time
        if wait_time > 0:
            time.sleep(wait_time)
env.close()
