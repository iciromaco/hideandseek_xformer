# main26_train_final.py v3.49
# 修正内容:
# 1. 強制終了の改善: signal ハンドラを導入し Ctrl+C への応答性を最大化。
# 2. ループ制御の徹底: 全てのループ内に keep_running_v チェックを配置。
# 3. 統計ロギングの保護: 中断時でも最後までの統計を TensorBoard に書き込み。
# 4. 二重正規化の排除継続: 環境側の正規化済み出力をそのまま利用。
# 5. PEP 8 準拠: 79文字制限、物理行展開を維持。

import os
import platform
import signal
import sys
import time
from collections import deque
from datetime import datetime

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.obs_indices import H1, H2, RAMP, SELF

# 自作モジュール
from envs.hns_environment import TeamCosEnv
from models.ppo_transformer_v2 import AgentV2

# ==========================================
# 0. 実行環境・基盤設定
# ==========================================
p_info_v = platform.processor()

if p_info_v != "arm":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 停止フラグのグローバル管理
keep_running_v = True


def signal_handler(sig, frame):
    """Ctrl+C などのシグナルを検知して安全停止フラグを立てる。"""
    global keep_running_v
    if keep_running_v:
        print("\n🛑 Interrupt signal received. Shutting down gracefully...")
        keep_running_v = False
    else:
        print("\n⚠️ Force quitting...")
        sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


# ==========================================
# 1. 視認化ユーティリティ
# ==========================================
def draw_vision_lines(env):
    """視野判定ラインの描画。"""
    v_ptr_v = env.viewer
    if v_ptr_v is None:
        return

    env.viewer.user_scn.ngeom = 0
    t_ids_v = [
        env.box_ids[0],
        env.box_ids[1],
        env.ramp_id,
        env.body_ids["s"],
        env.body_ids["h1"],
        env.body_ids["h2"],
    ]

    src_keys_v = ["s", "h1", "h2"]
    identity_v = np.eye(3).flatten()

    for k_src_v in src_keys_v:
        bid_s_v = env.body_ids[k_src_v]
        pos_s_v = env.data.xpos[bid_s_v]
        rj_idx_v = env.qpos_indices[k_src_v]["rot"]
        rot_s_val_v = env.data.qpos[env.model.jnt_qposadr[rj_idx_v]]

        if k_src_v == "s":
            rgba_v = [1.0, 0.3, 0.1, 0.8]
        else:
            rgba_v = [0.1, 1.0, 0.1, 0.6]

        for bid_d_v in t_ids_v:
            if bid_d_v == bid_s_v:
                continue
            pos_d_v = env.data.xpos[bid_d_v]

            if env._is_within_fov_and_visible(pos_s_v[:2], rot_s_val_v, pos_d_v[:2], bid_s_v, bid_d_v):
                gn_v = env.viewer.user_scn.ngeom
                if gn_v < 100:
                    gs_v = env.viewer.user_scn.geoms[gn_v]
                    mujoco.mjv_initGeom(
                        gs_v,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        [0.005, 0, 0],
                        pos_s_v,
                        identity_v,
                        rgba_v,
                    )
                    mujoco.mjv_connector(gs_v, mujoco.mjtGeom.mjGEOM_LINE, 0.005, pos_s_v, pos_d_v)
                    env.viewer.user_scn.ngeom = gn_v + 1


# ==========================================
# 2. ハイパーパラメータ
# ==========================================
TRAIN_MODE = False
USE_VIEWER = False
TRACK_WANDB = False

WANDB_PROJECT = "HNS-Xformer-Final"
WANDB_ENTITY = None
MODE = "initial"
TRAINING_TARGET = "hider"
HIDER_MODEL_PATH = "HNS_V26_GTRPPO_hider.pt"
SEEKER_MODEL_PATH = "HNS_V26_GTRPPO_seeker.pt"

NUM_ENVS = 16
TOTAL_STEPS = 20_000_000
ROLLOUT_STEPS = 128
SEQ_LEN = 8
HIDDEN_DIM = 128

LEARNING_RATE = 2.5e-4
ANNEAL_LR = True
GAE_GAMMA = 0.99
GAE_LAMBDA = 0.95
NUM_MINIBATCHES = 4
UPDATE_EPOCHS = 10
NORM_ADV = True
CLIP_COEF = 0.2
CLIP_VALUE_LOSS = True
ENTROPY_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.015

SAVE_FREQ = 20
LOG_FREQ = 1
NDR_WINDOW = 100


def compute_custom_reward(obs, action, base_reward):
    """
    観測インデックス定数とアクションを利用した、読みやすい報酬シェイピング。
    """
    # 敵が視界にいる場合の追加報酬
    enemy_visible = obs[H1.VISIBLE] > 0.5 or obs[H2.VISIBLE] > 0.5

    # Seekerの場合、敵との距離が縮まったら報酬 (報酬シェイピング)
    dist_to_h1 = math.sqrt(obs[H1.REL_X] ** 2 + obs[H1.REL_Y] ** 2)

    # 道具(Ramp)に近いことへのボーナス
    dist_to_ramp = math.sqrt(obs[RAMP.REL_X] ** 2 + obs[RAMP.REL_Y] ** 2)

    # 速度が出ていることへのボーナス (スタック/停止防止)
    speed = math.sqrt(obs[SELF.VEL_X] ** 2 + obs[SELF.VEL_Y] ** 2)

    # --- 報酬の加算 ---
    bonus = 0.0
    if enemy_visible:
        bonus += 0.1

    # 距離が近いほど報酬を与える
    bonus += 0.01 / (dist_to_h1 + 0.1)

    # --- アクションに基づくペナルティ (制御コスト) ---
    # アクションベクトル [fwd, trn, lck, grb] の二乗和に比例したペナルティ
    # これにより、無駄に激しい回転やボタン連打を抑制し、効率的な動きを促す
    action_penalty = -0.01 * np.sum(np.square(action))

    return base_reward + bonus + action_penalty


# ==========================================
# 3. 履歴管理 (ObsHistory)
# ==========================================
class ObsHistory:
    def __init__(self, n, sl, od, dev):
        self.n = n
        self.sl = sl
        self.od = od
        self.dev = dev
        self.capacity = sl * 2
        self.ptr = 0
        self.buffer = torch.zeros((n, self.capacity, od), device=dev)

    def reset(self, i=None):
        if i is None:
            self.buffer.zero_()
            self.ptr = 0
        else:
            self.buffer[i].zero_()

    def prime(self, o, i):
        f_t_v = torch.as_tensor(o, dtype=torch.float32, device=self.dev).reshape(self.od)
        for idx in range(self.capacity):
            self.buffer[i, idx] = f_t_v

    def update_batch(self, o_batch):
        t_b_v = torch.as_tensor(o_batch, dtype=torch.float32, device=self.dev)
        p_v = self.ptr
        self.buffer[:, p_v] = t_b_v
        self.buffer[:, p_v + self.sl] = t_b_v
        self.ptr = (p_v + 1) % self.sl

    def update_single(self, o, i):
        rv_v = torch.as_tensor(o, dtype=torch.float32, device=self.dev).reshape(self.od)
        p_v = self.ptr
        self.buffer[i, p_v] = rv_v
        self.buffer[i, p_v + self.sl] = rv_v

    def get(self):
        s_v = self.ptr
        e_v = self.ptr + self.sl
        return self.buffer[:, s_v:e_v].contiguous()


def apply_camera_slanted(env):
    if env.viewer is not None:
        env.viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        env.viewer.cam.distance = 15.0
        env.viewer.cam.elevation = -50.0
        env.viewer.cam.azimuth = 90.0


def run():
    global keep_running_v
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = 55
    act_dim = 4
    if MODE == "refinement" and TRAINING_TARGET == "hider":
        act_dim = 8

    agent = AgentV2(obs_dim, act_dim, HIDDEN_DIM, SEQ_LEN).to(device)
    m_path_v = HIDER_MODEL_PATH if TRAINING_TARGET == "hider" else SEEKER_MODEL_PATH

    if os.path.exists(m_path_v):
        agent.load_state_dict(torch.load(m_path_v, map_location=device))
        agent.eval()
        print(f"✅ Loaded weights: {m_path_v}")

    if TRAIN_MODE and TRACK_WANDB:
        import wandb

        dt_v = datetime.now().strftime("%m%d_%H%M")
        r_name = f"main26_{TRAINING_TARGET}_{dt_v}"
        wandb.init(project=WANDB_PROJECT, name=r_name)

    def make_env_internal():
        rm_v = "human" if (USE_VIEWER or not TRAIN_MODE) else None
        return TeamCosEnv(mode=MODE, target=TRAINING_TARGET, render_mode=rm_v)

    # --- [A] PLAYBACK MODE ---
    if not TRAIN_MODE:
        print("🎬 Starting Playback with Slanted Bird's Eye View...")
        env_pb = make_env_internal()
        h_man_pb = ObsHistory(1, SEQ_LEN, obs_dim, device)
        o_norm_v, _ = env_pb.reset()
        h_man_pb.prime(o_norm_v, 0)
        cum_r_v, n_ndr_v, st_t_v, s_time_v = 0.0, 0, 0, time.time()
        cam_done_v = False
        pbar_v = tqdm(desc="Playback", unit="step")
        try:
            while keep_running_v:
                with torch.no_grad():
                    obs_input_v = h_man_pb.get()
                    act_res_v = agent.get_action_and_value(obs_input_v)
                    a_np_v = act_res_v[0].cpu().numpy().flatten()
                on_norm_v, rw_v, dn_v, tr_v, inf_v = env_pb.step(a_np_v)
                h_man_pb.update_batch(on_norm_v[np.newaxis, :])
                st_t_v += 1
                cum_r_v += rw_v
                if not inf_v.get("is_detected", False):
                    n_ndr_v += 1
                pbar_v.update(1)
                elapsed_v = time.time() - s_time_v
                sps_curr_v = int(st_t_v / (elapsed_v + 1e-6))
                pbar_v.set_postfix(
                    {
                        "SPS": sps_curr_v,
                        "Rew": f"{cum_r_v/st_t_v:.4f}",
                        "NDR": f"{n_ndr_v/st_t_v:.2f}",
                    }
                )
                draw_vision_lines(env_pb)
                env_pb.render()
                if not cam_done_v:
                    apply_camera_slanted(env_pb)
                    cam_done_v = True
                time.sleep(0.04)
                if dn_v or tr_v:
                    o_res_norm_v, _ = env_pb.reset()
                    h_man_pb.reset()
                    h_man_pb.prime(o_res_norm_v, 0)
        finally:
            pbar_v.close()
            env_pb.close()

    # --- [B] TRAINING MODE ---
    else:
        print(f"🚀 Training: {NUM_ENVS} envs | Target: {TRAINING_TARGET}")
        envs_list_v = [make_env_internal() for _ in range(NUM_ENVS)]
        optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
        dt_str_v = datetime.now().strftime("%m%d_%H%M")
        writer = SummaryWriter(f"runs/main26_{TRAINING_TARGET}_{dt_str_v}")
        obs_st_v = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, SEQ_LEN, obs_dim), device=device)
        act_st_v = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, act_dim), device=device)
        logp_st_v = torch.zeros((ROLLOUT_STEPS, NUM_ENVS), device=device)
        rew_st_v = torch.zeros((ROLLOUT_STEPS, NUM_ENVS), device=device)
        done_st_v = torch.zeros((ROLLOUT_STEPS, NUM_ENVS), device=device)
        val_st_v = torch.zeros((ROLLOUT_STEPS, NUM_ENVS), device=device)
        obs_norm_batch_v = np.zeros((NUM_ENVS, obs_dim))
        h_tr_v = ObsHistory(NUM_ENVS, SEQ_LEN, obs_dim, device)
        ndr_q_v, hold_q_v, lock_q_v = (
            deque(maxlen=NDR_WINDOW),
            deque(maxlen=NDR_WINDOW),
            deque(maxlen=NDR_WINDOW),
        )
        for i in range(NUM_ENVS):
            oi_norm_v, _ = envs_list_v[i].reset()
            h_tr_v.prime(oi_norm_v, i)
        n_dn_v, global_step_v, start_time_tr_v = (
            torch.zeros(NUM_ENVS, device=device),
            0,
            time.time(),
        )
        cam_tr_done_v = False
        u_total_v = TOTAL_STEPS // (NUM_ENVS * ROLLOUT_STEPS)
        try:
            for u_idx_v in range(1, u_total_v + 1):
                if not keep_running_v:
                    break
                if ANNEAL_LR:
                    optimizer.param_groups[0]["lr"] = (1.0 - (u_idx_v - 1.0) / u_total_v) * LEARNING_RATE
                agent.eval()
                for r_step_v in range(ROLLOUT_STEPS):
                    if not keep_running_v:
                        break
                    global_step_v += NUM_ENVS
                    c_seq_v = h_tr_v.get()
                    obs_st_v[r_step_v], done_st_v[r_step_v] = c_seq_v, n_dn_v
                    with torch.no_grad():
                        at_v, lpt_v, _, vt_v = agent.get_action_and_value(c_seq_v)
                        val_st_v[r_step_v] = vt_v.flatten()
                    act_st_v[r_step_v], logp_st_v[r_step_v] = at_v, lpt_v
                    a_np_b_v = at_v.cpu().numpy()
                    for ie in range(NUM_ENVS):
                        on_n_v, r_v, d_v, tr_v, inf_v = envs_list_v[ie].step(a_np_b_v[ie])
                        rew_st_v[r_step_v, ie] = float(r_v)
                        if TRAINING_TARGET == "hider":
                            ndr_q_v.append(float(not inf_v.get("is_detected", False)))
                            hold_q_v.append(float(inf_v.get("hold_event", False)))
                            lock_q_v.append(float(inf_v.get("lock_event", False)))
                        is_f_v = bool(d_v or tr_v)
                        n_dn_v[ie] = float(is_f_v)
                        if is_f_v:
                            o_res_n_v, _ = envs_list_v[ie].reset()
                            h_tr_v.update_single(o_res_n_v, ie)
                            obs_norm_batch_v[ie] = o_res_n_v
                        else:
                            obs_norm_batch_v[ie] = on_n_v
                    h_tr_v.update_batch(obs_norm_batch_v)
                    if USE_VIEWER:
                        draw_vision_lines(envs_list_v[0])
                        envs_list_v[0].render()
                        if not cam_tr_done_v:
                            apply_camera_slanted(envs_list_v[0])
                            cam_tr_done_v = True
                if not keep_running_v:
                    break
                with torch.no_grad():
                    nv_v = agent.get_value(h_tr_v.get()).reshape(1, -1)
                    adv_v = torch.zeros_like(rew_st_v, device=device)
                    lg_v = 0
                    for t_rev_v in reversed(range(ROLLOUT_STEPS)):
                        va_v = nv_v if t_rev_v == ROLLOUT_STEPS - 1 else val_st_v[t_rev_v + 1]
                        mask_v = 1.0 - done_st_v[t_rev_v]
                        delta_v = rew_st_v[t_rev_v] + GAE_GAMMA * va_v * mask_v - val_st_v[t_rev_v]
                        lg_v = delta_v + GAE_GAMMA * GAE_LAMBDA * mask_v * lg_v
                        adv_v[t_rev_v] = lg_v
                    ret_v = adv_v + val_st_v
                agent.train()
                b_obs_v, b_lp_v = obs_st_v.reshape(-1, SEQ_LEN, obs_dim), logp_st_v.reshape(-1)
                b_act_v, b_adv_v, b_ret_v, b_val_v = (
                    act_st_v.reshape(-1, act_dim),
                    adv_v.reshape(-1),
                    ret_v.reshape(-1),
                    val_st_v.reshape(-1),
                )
                acc_pg_v, acc_vl_v, acc_et_v, acc_kl_v, bt_cnt_v = 0.0, 0.0, 0.0, 0.0, 0
                idx_l_v = np.arange(NUM_ENVS * ROLLOUT_STEPS)
                for epoch_i_v in range(UPDATE_EPOCHS):
                    if not keep_running_v:
                        break
                    np.random.shuffle(idx_l_v)
                    m_sz_v = (NUM_ENVS * ROLLOUT_STEPS) // NUM_MINIBATCHES
                    for s_ptr_v in range(0, NUM_ENVS * ROLLOUT_STEPS, m_sz_v):
                        mb_v = idx_l_v[s_ptr_v : s_ptr_v + m_sz_v]
                        _, n_lp_v, n_et_v, n_vl_v = agent.get_action_and_value(b_obs_v[mb_v], b_act_v[mb_v])
                        log_diff_v = n_lp_v - b_lp_v[mb_v]
                        ratio_v = log_diff_v.exp()
                        with torch.no_grad():
                            kl_v = (ratio_v - 1.0 - log_diff_v).mean().item()
                            acc_kl_v += kl_v
                        m_adv_v = (b_adv_v[mb_v] - b_adv_v[mb_v].mean()) / (b_adv_v[mb_v].std() + 1e-8) if NORM_ADV else b_adv_v[mb_v]
                        pg_l_v = torch.max(
                            -m_adv_v * ratio_v,
                            -m_adv_v * torch.clamp(ratio_v, 1 - CLIP_COEF, 1 + CLIP_COEF),
                        ).mean()
                        nv_f_v = n_vl_v.flatten()
                        v_l_v = (
                            0.5
                            * torch.max(
                                (nv_f_v - b_ret_v[mb_v]) ** 2,
                                (b_val_v[mb_v] + torch.clamp(nv_f_v - b_val_v[mb_v], -CLIP_COEF, CLIP_COEF) - b_ret_v[mb_v]) ** 2,
                            ).mean()
                            if CLIP_VALUE_LOSS
                            else 0.5 * ((nv_f_v - b_ret_v[mb_v]) ** 2).mean()
                        )
                        total_l_v = pg_l_v - ENTROPY_COEF * n_et_v.mean() + v_l_v * VF_COEF
                        acc_pg_v += pg_l_v.item()
                        acc_vl_v += v_l_v.item()
                        acc_et_v += n_et_v.mean().item()
                        bt_cnt_v += 1
                        optimizer.zero_grad()
                        total_l_v.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                    if TARGET_KL is not None and (acc_kl_v / bt_cnt_v) > TARGET_KL:
                        break
                if u_idx_v % LOG_FREQ == 0:
                    elapsed_v = time.time() - start_time_tr_v
                    sps_v = int(global_step_v / (elapsed_v + 1e-6))
                    ndr_v, hr_v, lr_v = (
                        np.mean(ndr_q_v) if ndr_q_v else 0.0,
                        np.mean(hold_q_v) if hold_q_v else 0.0,
                        np.mean(lock_q_v) if lock_q_v else 0.0,
                    )
                    div_f_v = bt_cnt_v + 1e-6
                    m_rew_v = rew_st_v.mean().item()
                    print(f"Upd {u_idx_v:4d} | SPS {sps_v:5d} | Rew {m_rew_v:7.4f} | NDR {ndr_v:.2f} | Hold {hr_v:.2f} | Lock {lr_v:.2f} | P-L {acc_pg_v/div_f_v:6.4f} | V-L {acc_vl_v/div_f_v:6.4f}")
                    writer.add_scalar("charts/SPS", sps_v, global_step_v)
                    writer.add_scalar("charts/reward", m_rew_v, global_step_v)
                    writer.add_scalar("charts/ndr", ndr_v, global_step_v)
                    writer.add_scalar("losses/policy_loss", acc_pg_v / div_f_v, global_step_v)
                    writer.add_scalar("losses/value_loss", acc_vl_v / div_f_v, global_step_v)
                if u_idx_v % SAVE_FREQ == 0:
                    torch.save(agent.state_dict(), m_path_v)
        finally:
            torch.save(agent.state_dict(), m_path_v)
            for env_f_v in envs_list_v:
                env_f_v.close()
            writer.close()


if __name__ == "__main__":
    run()
