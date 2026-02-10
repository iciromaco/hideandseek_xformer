# main23_runner_cos.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
# 
# 【修正内容 (v25.25 - 統計出力の完全復旧と安定化)】
# 1. 統計収集ロジックの修正: terminated=False, truncated=True に戻し、ラッパーの挙動を安定化。
# 2. 統計抽出の冗長化: info["final_info"] と info["episode"] の両方を走査するオリジナル構造を維持。
# 3. ログ出力の強制化: update == 1 (初回) は必ず出力し、以降 10 ごとに確実に print。
# 4. 高速化ロジック完備: math置換、Lidarスキップ、キャッシュ、ミラーリングバッファを全て継承。

import os
import sys
import platform
import json
import time
import math
import numpy as np
import multiprocessing
from tqdm import tqdm

# --- 実行環境の最適化 ---
if platform.processor() != 'arm':
    for k in ["OMP", "MKL", "OPENBLAS", "VECLIB", "NUMEXPR"]:
        os.environ[f"{k}_NUM_THREADS"] = "1"

current_file_path = os.path.abspath(__file__)
search_path = os.path.dirname(current_file_path)

for _ in range(5):
    if os.path.exists(os.path.join(search_path, "main18_optimization.py")):
        if search_path not in sys.path:
            sys.path.insert(0, search_path)
        break
    search_path = os.path.dirname(search_path)

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# ==========================================
# 1. 実験設定
# ==========================================
MODE = "initial" 
EXPERIMENT_BASE_NAME = "HideAndSeek_Layer23_TeamCos"
TRAIN_TARGET = "HIDER" 
EXPERIMENT_NAME = f"{EXPERIMENT_BASE_NAME}_{MODE}"
LOAD_EXISTING_MODELS = True if MODE == "refinement" else False
EXECUTION_MODE =  "TRAIN" #"PLAY"
SAVE_MODEL = True
TRACK_WANDB = True           
FIXED_SEED = None
TRIAL_MODE = False

TOTAL_TIMESTEPS = 5000000 
NUM_ENVS = 8
NUM_STEPS = 300
LEARNING_RATE = 2e-4
ENT_COEF = 0.001
MINIBATCH_SIZE = 128
UPDATE_EPOCHS = 4

ACTION_REPEAT = 16          
PREP_STEPS = 80             
MAX_STEPS = 300             
FOV_DEG = 135               
TRANSFORMER_SEQ_LEN = 8

LIDAR_CACHE_POS_THRESH_SQ = 0.05**2
LIDAR_CACHE_ANG_THRESH = np.deg2rad(2.0)
RAYCAST_CACHE_POS_THRESH_SQ = 0.05**2

REWARD_HIDDEN_BONUS = 1.0  
COS_PENALTY_SCALE = 2.0    
REWARD_DISTANCE_DIFF_SCALE = 1.0 
PENALTY_SAFEGUARD = -20.0  

HIDER_THRUST_LIMIT = 0.40  
SEEKER_THRUST_LIMIT = 0.35 
SEEKER_RB_THRUST = 0.38 
SEEKER_RB_TURN_THRESH = math.pi/6 

SAVE_MODEL_PATH = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}.pt"

# ==========================================
# 2. 高速化データ構造
# ==========================================
class ObsHistory:
    def __init__(self, num_envs, seq_len, obs_dim, device):
        import torch
        self.buffer = torch.zeros((num_envs, seq_len * 2, obs_dim), device=device)
        self.device = device
        self.seq_len = seq_len
        self.ptr = 0

    def reset(self):
        self.buffer.zero_()
        self.ptr = 0

    def update(self, obs):
        import torch
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        self.buffer[:, self.ptr] = obs_tensor
        self.buffer[:, self.ptr + self.seq_len] = obs_tensor
        self.ptr = (self.ptr + 1) % self.seq_len

    def get(self):
        return self.buffer[:, self.ptr : self.ptr + self.seq_len]

def load_model_safely(model_obj, base_name, target_type):
    import torch
    candidates = [
        f"{base_name}_refinement_{target_type}.pt",
        f"{base_name}_initial_{target_type}.pt",
        f"{base_name}_{target_type}.pt"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                model_obj.load_state_dict(torch.load(path, map_location="cpu"))
                model_obj.eval()
                return path
            except Exception:
                continue
    return None

# ==========================================
# 3. 環境作成用ファクトリ
# ==========================================
def create_env(render_mode=None):
    import torch
    import mujoco
    import gymnasium as gym
    import main18_optimization as base_config
    from main18_optimization import Agent

    class TeamCosEnv(base_config.HideAndSeekEnv):
        def __init__(self, render_mode=None):
            super().__init__(render_mode=render_mode)
            cpu_dev = torch.device("cpu")
            self.npc_obs_history = {
                0: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev),
                1: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev),
                2: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
            }
            self.visible_cache = {0: {}, 1: {}, 2: {}}
            self.lidar_array_cache = {} 
            self.raycast_cache = {} 
            self._obs_memo = {}
            self.hidden_steps_count = 0
            self.caught_steps_count = 0
            self.prev_dist = {1: 0.0, 2: 0.0}
            self.s0_recovery_turn_dir = 1.0
            
            self.npc_hider_agent = Agent(53, 4).to("cpu")
            self.npc_seeker_agent = Agent(53, 4).to("cpu")
            
            should_print = (os.environ.get("NPC_MODELS_LOGGED") != "TRUE")
            load_model_safely(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER")
            load_model_safely(self.npc_seeker_agent, EXPERIMENT_BASE_NAME, "SEEKER")
            if should_print:
                os.environ["NPC_MODELS_LOGGED"] = "TRUE"

        def reset(self, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            self.hidden_steps_count = 0
            self.caught_steps_count = 0
            self._obs_memo.clear()
            self.lidar_array_cache.clear()
            self.raycast_cache.clear()
            
            sp_pos = self.data.xpos[self.s0_body][:2]
            for i in [1, 2]:
                bid = self.h1_body if i == 1 else self.h2_body
                diff = self.data.xpos[bid][:2] - sp_pos
                self.prev_dist[i] = math.sqrt(sum(diff**2))
            return obs, info

        def _get_cached_ray(self, agent_id, origin_p, direction, beam_id):
            angle = math.atan2(direction[1], direction[0])
            cache_key = (agent_id, beam_id)
            if cache_key in self.raycast_cache:
                c_hp, c_a, c_res, c_gid = self.raycast_cache[cache_key]
                dist_sq = (origin_p[0] - c_hp[0])**2 + (origin_p[1] - c_hp[1])**2
                if dist_sq < RAYCAST_CACHE_POS_THRESH_SQ:
                    ang_diff = (angle - c_a + math.pi) % (2.0 * math.pi) - math.pi
                    if abs(ang_diff) < 0.05: return c_res, c_gid
            gid = np.zeros(1, dtype=np.int32)
            from_p = np.array([origin_p[0], origin_p[1], 0.5], dtype=np.float64)
            dir_3d = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
            exclude = self.s0_body if agent_id == 0 else (self.h1_body if agent_id == 1 else self.h2_body)
            res = mujoco.mj_ray(self.model, self.data, from_p, dir_3d, None, 1, exclude, gid)
            self.raycast_cache[cache_key] = (origin_p.copy(), angle, res, gid[0])
            return res, gid[0]

        def _get_obs(self, agent_id, skip_lidar=False):
            if agent_id in self._obs_memo: return self._obs_memo[agent_id]
            if agent_id == 0: b_id, p_pref = self.s0_body, 's'
            elif agent_id == 1: b_id, p_pref = self.h1_body, 'h1'
            else: b_id, p_pref = self.h2_body, 'h2'
            hp = self.data.xpos[b_id][:2]
            j_id = self.model.joint(f'{p_pref}_rot').id
            hra = self.data.qpos[self.model.jnt_qposadr[j_id]]
            c_hra, s_hra = math.cos(hra), math.sin(hra)
            c_mhra, s_mhra = math.cos(-hra), math.sin(-hra)
            rot_mat = np.array([[c_mhra, -s_mhra], [s_mhra, c_mhra]])
            jx_id = self.model.joint(f'{p_pref}_x').id
            dof_adr = self.model.jnt_dofadr[jx_id]
            h_raw_vel = self.data.qvel[dof_adr : dof_adr + 2]
            h_obs_vel = (rot_mat @ h_raw_vel) / 12.0
            self_state = np.array([h_obs_vel[0], h_obs_vel[1], hra, c_hra, s_hra], dtype=np.float32)
            lidar_data = None
            if not skip_lidar:
                l_cache = self.lidar_array_cache.get(agent_id)
                if l_cache is not None:
                    dist_sq = (hp[0] - l_cache[0][0])**2 + (hp[1] - l_cache[0][1])**2
                    if dist_sq < LIDAR_CACHE_POS_THRESH_SQ:
                        ang_diff = (hra - l_cache[1] + math.pi) % (2.0 * math.pi) - math.pi
                        if abs(ang_diff) < LIDAR_CACHE_ANG_THRESH: lidar_data = l_cache[2]
                if lidar_data is None:
                    lidar_data = np.zeros(len(self.lidar_angles), dtype=np.float32)
                    for i, angle_offset in enumerate(self.lidar_angles):
                        beam_dir = angle_offset + hra
                        direction = np.array([math.cos(beam_dir), math.sin(beam_dir)])
                        dist, _ = self._get_cached_ray(agent_id, hp, direction, i + 100)
                        lidar_data[i] = min(dist, 2.5) / 2.5 if dist != -1 else 1.0
                    self.lidar_array_cache[agent_id] = (hp.copy(), hra, lidar_data.copy())
            else: lidar_data = np.zeros(len(self.lidar_angles), dtype=np.float32)
            my_vis = self.visible_cache[agent_id]
            my_vis.clear()
            targets = [self.box1_body, self.box2_body, self.ramp_body, self.h1_body, self.h2_body, self.s0_body]
            for tid in [t for t in targets if t != b_id]:
                tp_pos = self.data.xpos[tid]
                diff = tp_pos[:2] - hp
                dist_obj = math.sqrt(diff[0]**2 + diff[1]**2)
                target_ang = math.atan2(diff[1], diff[0])
                angle_rel = (target_ang - hra + math.pi) % (2.0 * math.pi) - math.pi
                if abs(angle_rel) > math.radians(FOV_DEG / 2.0):
                    my_vis[tid] = False
                    continue
                res, hit_gid = self._get_cached_ray(agent_id, hp, diff / (dist_obj + 1e-8), tid)
                hit_body = self.model.geom_bodyid[hit_gid] if res != -1 else -1
                my_vis[tid] = (hit_body == tid) or (res != -1 and res > dist_obj - 0.4)
            def get_rel_info(target_id, lock=None):
                if my_vis.get(target_id, False):
                    t_pos = self.data.xpos[target_id]
                    rel_p = rot_mat @ (t_pos[:2] - hp) / 12.0
                    q_rot = self.data.xquat[target_id]
                    yaw = math.atan2(2 * (q_rot[0]*q_rot[3] + q_rot[1]*q_rot[2]), 1 - 2 * (q_rot[2]**2 + q_rot[3]**2))
                    b_jnt = self.model.body_jntadr[target_id]
                    t_vel = self.data.qvel[b_jnt : b_jnt + 2] if b_jnt != -1 else np.zeros(2)
                    rel_v = rot_mat @ (t_vel - h_raw_vel) / 12.0
                    info = [rel_p[0], rel_p[1], rel_v[0], rel_v[1], math.cos(yaw - hra), math.sin(yaw - hra)]
                    if lock is not None: info.append(1.0 if lock else 0.0)
                    info.append(1.0)
                    return np.array(info, dtype=np.float32)
                return np.zeros(8 if lock is not None else 7, dtype=np.float32)
            if agent_id == 0:
                obs_out = np.concatenate([self_state, lidar_data, get_rel_info(self.box1_body, self.locked_boxes[self.box1_body]), get_rel_info(self.box2_body, self.locked_boxes[self.box2_body]), get_rel_info(self.ramp_body), get_rel_info(self.h1_body)[:5], get_rel_info(self.h2_body)[:5], np.zeros(3, dtype=np.float32)])
            else:
                partner_id = self.h2_body if agent_id == 1 else self.h1_body
                obs_out = np.concatenate([self_state, lidar_data, get_rel_info(self.box1_body, self.locked_boxes[self.box1_body]), get_rel_info(self.box2_body, self.locked_boxes[self.box2_body]), get_rel_info(self.ramp_body), get_rel_info(self.s0_body)[:5], get_rel_info(partner_id), [1.0 if self.grasping[agent_id] else 0.0]])
            self._obs_memo[agent_id] = obs_out.astype(np.float32)
            return self._obs_memo[agent_id]

        def _update_seeker_state(self):
            sp_pos = self.data.xpos[self.s0_body][:2]
            self._get_obs(0)
            v1, v2 = self.visible_cache[0].get(self.h1_body, False), self.visible_cache[0].get(self.h2_body, False)
            if v1 or v2:
                target_bid = self.h1_body if v1 else self.h2_body
                self.seeker_target_pos = self.data.xpos[target_bid][:2].copy()
                self.seeker_last_known_pos = self.seeker_target_pos.copy()
            elif self.seeker_last_known_pos is not None:
                diff = sp_pos - self.seeker_last_known_pos
                dist_to_mem = math.sqrt(sum(diff**2))
                if dist_to_mem < 0.5:
                    self.seeker_last_known_pos = None
                    self.seeker_search_timer = 50
                else: self.seeker_target_pos = self.seeker_last_known_pos.copy()
            else:
                if self.seeker_search_timer <= 0:
                    self.seeker_random_target, self.seeker_search_timer = self.np_random.uniform(-4, 4, 2), 80
                self.seeker_search_timer -= 1
                self.seeker_target_pos = self.seeker_random_target.copy()

        def _seeker_rule_based_policy(self):
            if self.current_step < PREP_STEPS: return 0.0, 0.0
            sp_pos, sr_val = self.data.xpos[self.s0_body][:2], self.data.qpos[self.srot_adr]
            lidar, obs_potential = self.lidar_array_cache[0][2], np.zeros(2)
            for i, ang in enumerate(self.lidar_angles):
                dist = lidar[i] * 2.5
                if dist < 1.0:
                    force = (1.0 - dist) / (dist + 0.1)
                    vec_ang = ang + sr_val
                    obs_potential[0] -= math.cos(vec_ang) * force
                    obs_potential[1] -= math.sin(vec_ang) * force
            diff = self.seeker_target_pos - sp_pos
            target_dir = diff / (math.sqrt(diff[0]**2 + diff[1]**2) + 1e-8)
            final_dir = target_dir + obs_potential * 1.5
            angle_diff = (math.atan2(final_dir[1], final_dir[0]) - sr_val + math.pi) % (2.0 * math.pi) - math.pi
            thrust = SEEKER_RB_THRUST
            if abs(angle_diff) > SEEKER_RB_TURN_THRESH: thrust *= 0.3
            sx_id = self.model.joint('s_x').id
            sx_adr = self.model.jnt_dofadr[sx_id]
            s_vel = math.sqrt(sum(self.data.qvel[sx_adr : sx_adr + 2]**2))
            if thrust > 0.05 and s_vel < 0.05: self.s0_stuck_timer += 5
            else: self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
            if self.s0_stuck_timer > 15:
                self.s0_recovery_mode, self.s0_stuck_timer, self.s0_recovery_turn_dir = 15, 0, self.np_random.choice([-1.0, 1.0])
            if self.s0_recovery_mode > 0:
                self.s0_recovery_mode -= 1
                return -0.2, 1.5 * self.s0_recovery_turn_dir
            return float(thrust), float(np.clip(angle_diff * 6.0, -3.0, 3.0))

        def step(self, action):
            self._obs_memo.clear()
            self.current_step += 1
            for i in [1, 2]: self.lock_cooldown[i] = max(0, self.lock_cooldown[i] - 1)
            self._update_seeker_state()
            self.data.ctrl[:] = 0.0
            if TRAIN_TARGET == "HIDER":
                m_idx = self._apply_action(self.learning_agent_id, action)
                self.data.ctrl[m_idx] = float(action[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[m_idx + 1] = float(action[1])
                partner_id = 2 if self.learning_agent_id == 1 else 1
                act_p = self._get_npc_action(partner_id, "HIDER")
                p_idx = self._apply_action(partner_id, act_p)
                self.data.ctrl[p_idx] = float(act_p[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[p_idx + 1] = float(act_p[1])
                s_thr, s_turn = self._seeker_rule_based_policy()
                self.data.ctrl[0], self.data.ctrl[1] = s_thr, s_turn
            else:
                self.data.ctrl[0] = float(action[0]) * SEEKER_THRUST_LIMIT
                self.data.ctrl[1] = float(action[1])
                for i in [1, 2]:
                    act_n = self._get_npc_action(i, "HIDER")
                    n_idx = self._apply_action(i, act_n)
                    self.data.ctrl[n_idx] = float(act_n[0]) * HIDER_THRUST_LIMIT
                    self.data.ctrl[n_idx + 1] = float(act_n[1])
            for _ in range(ACTION_REPEAT):
                for b_id, pose in self.locked_pose.items():
                    if self.locked_boxes[b_id]:
                        jid = self.box1_joint_id if b_id == self.box1_body else self.box2_joint_id
                        q_pos, d_vel = self.model.jnt_qposadr[jid], self.model.jnt_dofadr[jid]
                        self.data.qpos[q_pos : q_pos + 7], self.data.qvel[d_vel : d_vel + 6] = pose, 0
                mujoco.mj_step(self.model, self.data)
            self._obs_memo.clear()
            obs_learner = self._get_obs(self.learning_agent_id)
            self._get_obs(0, skip_lidar=True)
            team_reward, l_body = 0.0, self.h1_body if self.learning_agent_id == 1 else self.h2_body
            if not self.visible_cache[0].get(l_body, False): self.hidden_steps_count += 1
            if any(self.visible_cache[0].get(b, False) for b in [self.h1_body, self.h2_body]): self.caught_steps_count += 1
            sp_xpos = self.data.xpos[self.s0_body][:2]
            for h_idx, b_id in [(1, self.h1_body), (2, self.h2_body)]:
                h_xpos = self.data.xpos[b_id][:2]
                diff = h_xpos - sp_xpos
                cur_d = math.sqrt(diff[0]**2 + diff[1]**2)
                if self.visible_cache[0].get(b_id, False):
                    sr = self.data.qpos[self.srot_adr]
                    dv_norm = diff / (cur_d + 1e-8)
                    cos_v = dv_norm[0] * math.cos(sr) + dv_norm[1] * math.sin(sr)
                    h_rew = -cos_v * COS_PENALTY_SCALE + (cur_d - self.prev_dist[h_idx]) * REWARD_DISTANCE_DIFF_SCALE
                else: h_rew = REWARD_HIDDEN_BONUS
                if abs(h_xpos[0]) > 6.5 or abs(h_xpos[1]) > 6.5: h_rew += PENALTY_SAFEGUARD
                team_reward, self.prev_dist[h_idx] = team_reward + h_rew, cur_d
            truncated = (self.current_step >= MAX_STEPS)
            stats = {"hidden_steps": float(self.hidden_steps_count), "caught_steps": float(self.caught_steps_count)}
            return obs_learner, float(team_reward if TRAIN_TARGET == "HIDER" else -team_reward), False, truncated, stats

        def _get_npc_action(self, agent_id, agent_type):
            import torch
            obs = self._get_obs(agent_id)
            self.npc_obs_history[agent_id].update(obs)
            model = self.npc_hider_agent if agent_type == "HIDER" else self.npc_seeker_agent
            if model:
                with torch.no_grad():
                    act, _, _, _ = model.get_action_and_value(self.npc_obs_history[agent_id].get())
                return act.cpu().numpy()[0]
            return self.action_space.sample() * 0.5

    return TeamCosEnv(render_mode=render_mode)

# ==========================================
# 4. メイン処理
# ==========================================
def main():
    if platform.system() == "Linux":
        try: multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError: pass
            
    if EXECUTION_MODE == "PLAY":
        import torch
        from main18_optimization import Agent
        env = create_env(render_mode="human")
        agent = Agent(env.observation_space.shape[0], env.action_space.shape[0]).to("cpu")
        load_model_safely(agent, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        agent.eval()
        o_hist = ObsHistory(1, TRANSFORMER_SEQ_LEN, env.observation_space.shape[0], "cpu")
        try:
            while True:
                next_obs, _ = env.reset()
                o_hist.reset()
                o_hist.update(next_obs)
                done = False
                while not done:
                    loop_start = time.time()
                    with torch.no_grad(): action, _, _, _ = agent.get_action_and_value(o_hist.get())
                    next_obs, reward, term, trunc, info = env.step(action.cpu().numpy()[0])
                    done = term or trunc
                    o_hist.update(next_obs)
                    env.render()
                    wait = (0.005 * ACTION_REPEAT) - (time.time() - loop_start)
                    if wait > 0: time.sleep(wait)
        except KeyboardInterrupt: pass
        finally: env.close()
        return

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    import wandb
    import main18_optimization as base_config
    from main18_optimization import Agent

    def env_factory():
        import gymnasium as gym
        return gym.wrappers.RecordEpisodeStatistics(create_env())
        
    import gymnasium as gym
    vec_envs = gym.vector.AsyncVectorEnv([env_factory for _ in range(NUM_ENVS)])
    device = torch.device("cuda" if torch.cuda.is_available() and base_config.CUDA else "cpu")
    run_name = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{int(time.time())}"
    if TRACK_WANDB:
        wandb.init(project=base_config.WANDB_PROJECT_NAME, config={"Target": TRAIN_TARGET, "v": "25.25_fast"}, name=run_name, sync_tensorboard=False, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    agent = Agent(vec_envs.single_observation_space.shape[0], vec_envs.single_action_space.shape[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    global_step = 0
    if LOAD_EXISTING_MODELS:
        lp = load_model_safely(agent, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if lp:
            cp = lp.replace('.pt', '_checkpoint.json')
            if os.path.exists(cp):
                with open(cp, 'r') as f: global_step = json.load(f).get('global_step', 0)

    initial_global_step = global_step
    obs_hist = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device)
    obs_b = torch.zeros((NUM_STEPS, NUM_ENVS, TRANSFORMER_SEQ_LEN, 53), device=device)
    actions_b = torch.zeros((NUM_STEPS, NUM_ENVS, 4), device=device)
    logprobs_b = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    rewards_b = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    dones_b = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    values_b = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    
    next_obs, _ = vec_envs.reset(seed=FIXED_SEED if FIXED_SEED else int(time.time()))
    next_done = torch.zeros(NUM_ENVS).to(device)
    obs_hist.update(next_obs)
    start_time = time.time()
    num_updates = int(max(1, (TOTAL_TIMESTEPS - global_step) // (NUM_ENVS * NUM_STEPS)))
    h_returns, h_hidden, h_caught = [], [], []

    try:
        for update in tqdm(range(1, num_updates + 1), desc="Updates"):
            for step in range(NUM_STEPS):
                global_step += NUM_ENVS
                obs_b[step], dones_b[step] = obs_hist.get(), next_done
                with torch.no_grad(): act, lp, _, val = agent.get_action_and_value(obs_hist.get())
                values_b[step] = val.flatten()
                actions_b[step], logprobs_b[step] = act, lp
                next_obs, rew, term, trunc, info = vec_envs.step(act.cpu().numpy())
                done_envs = np.logical_or(term, trunc)
                if "final_info" in info:
                    for i in range(NUM_ENVS):
                        if done_envs[i] and info["final_info"][i] is not None:
                            it = info["final_info"][i]
                            if "episode" in it: h_returns.append(float(it["episode"]["r"]))
                            if "hidden_steps" in it: h_hidden.append(float(it["hidden_steps"]))
                            if "caught_steps" in it: h_caught.append(float(it["caught_steps"]))
                elif "episode" in info:
                    mask = info.get("_episode", done_envs)
                    for i in range(NUM_ENVS):
                        if mask[i] and done_envs[i]:
                            h_returns.append(float(info["episode"]["r"][i]))
                            try:
                                if "hidden_steps" in info: h_hidden.append(float(info["hidden_steps"][i]))
                                if "caught_steps" in info: h_caught.append(float(info["caught_steps"][i]))
                            except Exception: pass
                rewards_b[step] = torch.tensor(rew).to(device).view(-1)
                next_done = torch.tensor(done_envs).to(device, dtype=torch.float32)
                obs_hist.update(next_obs)

            with torch.no_grad():
                next_val = agent.get_value(obs_hist.get()).reshape(1, -1)
                advs, lastgaelam = torch.zeros_like(rewards_b).to(device), 0
                for t in reversed(range(NUM_STEPS)):
                    nt = 1.0 - (next_done if t == NUM_STEPS - 1 else dones_b[t+1])
                    v_post = next_val if t == NUM_STEPS - 1 else values_b[t+1]
                    delta = rewards_b[t] + base_config.GAMMA * v_post * nt - values_b[t]
                    advs[t] = lastgaelam = delta + base_config.GAMMA * base_config.GAE_LAMBDA * nt * lastgaelam
                returns = advs + values_b

            f_obs, f_lp, f_act, f_adv, f_ret = obs_b.reshape(-1, TRANSFORMER_SEQ_LEN, 53), logprobs_b.reshape(-1), actions_b.reshape(-1, 4), advs.reshape(-1), returns.reshape(-1)
            for _ in range(UPDATE_EPOCHS):
                inds = np.random.permutation(NUM_STEPS * NUM_ENVS)
                for s in range(0, NUM_STEPS * NUM_ENVS, MINIBATCH_SIZE):
                    mb = inds[s : s + MINIBATCH_SIZE]
                    _, n_lp, ent, n_v = agent.get_action_and_value(f_obs[mb], f_act[mb])
                    ratio = (n_lp - f_lp[mb]).exp()
                    mb_adv = (f_adv[mb] - f_adv[mb].mean()) / (f_adv[mb].std() + 1e-8)
                    pg_l = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                    v_l = 0.5 * ((n_v.view(-1) - f_ret[mb]) ** 2).mean()
                    loss = pg_l - ENT_COEF * ent.mean() + 0.5 * v_l
                    optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 0.5); optimizer.step()

            # --- ログ出力ロジックの修正 ---
            if update == 1 or update % 10 == 0 or update == num_updates or TRIAL_MODE:
                dt = time.time() - start_time
                sps = int((global_step - initial_global_step) / dt) if dt > 0 else 0
                log_data = {"charts/SPS": sps, "losses/total_loss": loss.item(), "global_step": global_step}
                
                # tqdmのバーを壊さないように flush=True で出力
                if h_returns:
                    avg_r, avg_h, avg_c = np.mean(h_returns), np.mean(h_hidden), np.mean(h_caught)
                    log_data.update({"charts/episodic_return": avg_r, "charts/steps_hidden": avg_h, "charts/steps_caught": avg_c})
                    print(f"\nUpdate {update}, Step {global_step}, SPS {sps}, Ret {avg_r:.1f}, Hid {avg_h:.1f}", flush=True)
                    h_returns, h_hidden, h_caught = [], [] , []
                else:
                    print(f"\nUpdate {update}, Step {global_step}, SPS {sps} (Collecting stats...)", flush=True)
                
                if TRACK_WANDB: wandb.log(log_data)
                    
    except KeyboardInterrupt: print("Interrupted.")
    if SAVE_MODEL:
        torch.save(agent.state_dict(), SAVE_MODEL_PATH)
        with open(SAVE_MODEL_PATH.replace('.pt', '_checkpoint.json'), 'w') as f: json.dump({'global_step': global_step}, f)
    vec_envs.close(); writer.close()
    if TRACK_WANDB: wandb.finish()

if __name__ == "__main__":
    main()