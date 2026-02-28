# main26_train_final.py v3.40
# 修正内容:
# 1. PEP 8 準拠: インラインコメント、行長制限、セミコロン、空白ルールを修正。
# 2. 統計算出の完全記述: 損失項や NDR の平均化工程を独立行で解体維持。
# 3. 再生モードの復元: tqdm postfix に SPS, Reward, NDR を表示する形式を保持。
# 4. カメラ俯瞰角度の適用: Elevation -50.0 の斜め俯瞰設定。
# 5. エラー耐性向上: 未定義名や重複インポートを整理。

import os
import sys
import time
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mujoco
import mujoco.viewer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from collections import deque

# 自作モジュール
from envs.hns_environment import TeamCosEnv
from models.ppo_transformer_v2 import AgentV2

# ==========================================
# 0. 実行環境・基盤設定
# ==========================================
p_info = platform.processor()

if p_info != 'arm':
    # マルチスレッドによる演算競合を防止し、単一コアの SPS 効率を最大化
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ==========================================
# 1. 視認化ユーティリティ
# ==========================================
def draw_vision_lines(env):
    """視野判定ラインの描画工程。"""
    v_ptr = env.viewer
    if v_ptr is None:
        return

    env.viewer.user_scn.ngeom = 0
    t_ids = []
    t_ids.append(env.box_ids[0])
    t_ids.append(env.box_ids[1])
    t_ids.append(env.ramp_id)
    t_ids.append(env.body_ids["s"])
    t_ids.append(env.body_ids["h1"])
    t_ids.append(env.body_ids["h2"])

    src_keys = ["s", "h1", "h2"]

    for k_src in src_keys:
        bid_s = env.body_ids[k_src]
        pos_s = env.data.xpos[bid_s]
        rj = env.qpos_indices[k_src]['rot']
        rot_s = env.data.qpos[env.model.jnt_qposadr[rj]]

        rgba = [0.1, 1.0, 0.1, 0.6]  # Hider
        if k_src == "s":
            rgba = [1.0, 0.3, 0.1, 0.8]  # Seeker

        for bid_d in t_ids:
            if bid_d == bid_s:
                continue
            pos_d = env.data.xpos[bid_d]

            if env._is_within_fov_and_visible(pos_s[:2], rot_s, pos_d[:2],
                                              bid_s, bid_d):
                gn = env.viewer.user_scn.ngeom
                if gn < 100:
                    gs = env.viewer.user_scn.geoms[gn]
                    mujoco.mjv_initGeom(
                        gs, mujoco.mjtGeom.mjGEOM_LINE, [0.005, 0, 0],
                        pos_s, np.eye(3).flatten(), rgba
                    )
                    mujoco.mjv_connector(
                        gs, mujoco.mjtGeom.mjGEOM_LINE, 0.005, pos_s, pos_d
                    )
                    env.viewer.user_scn.ngeom = gn + 1


# ==========================================
# 2. 全ハイパーパラメータ ＆ 定数
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
        f = torch.as_tensor(o, dtype=torch.float32, device=self.dev).reshape(
            self.od
        )
        for idx in range(self.capacity):
            self.buffer[i, idx] = f

    def update_batch(self, o_batch):
        t = torch.as_tensor(o_batch, dtype=torch.float32, device=self.dev)
        p = self.ptr
        self.buffer[:, p] = t
        self.buffer[:, p + self.sl] = t
        self.ptr = (p + 1) % self.sl

    def update_single(self, o, i):
        rv = torch.as_tensor(o, dtype=torch.float32,
                             device=self.dev).reshape(self.od)
        p = self.ptr
        self.buffer[i, p] = rv
        self.buffer[i, p + self.sl] = rv

    def get(self):
        s = self.ptr
        e = self.ptr + self.sl
        return self.buffer[:, s:e].contiguous()


# ==========================================
# 4. 実行ユーティリティ
# ==========================================
def batch_normalize_obs(obs_batch):
    """NumPy ベクトル演算による高速正規化。"""
    res = obs_batch.copy()
    res[:, 0:2] /= 10.0
    res[:, 2] /= 5.0
    res[:, 5:17] /= 15.0
    res[:, 17:55] /= 12.0
    return res


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = 55
    act_dim = 4
    if MODE == "refinement" and TRAINING_TARGET == "hider":
        act_dim = 8

    agent = AgentV2(obs_dim, act_dim, HIDDEN_DIM, SEQ_LEN).to(device)
    m_path = HIDER_MODEL_PATH if TRAINING_TARGET == "hider" else \
        SEEKER_MODEL_PATH
    if os.path.exists(m_path):
        agent.load_state_dict(torch.load(m_path, map_location=device))
        agent.eval()
        print(f"✅ Loaded weights: {m_path}")

    if TRAIN_MODE and TRACK_WANDB:
        import wandb
        dt = datetime.now().strftime('%m%d_%H%M')
        r_name = f"main26_{TRAINING_TARGET}_{dt}"
        wandb.init(project=WANDB_PROJECT, name=r_name)

    def make_env_internal():
        rm = "human" if (USE_VIEWER or not TRAIN_MODE) else None
        return TeamCosEnv(mode=MODE, target=TRAINING_TARGET, render_mode=rm)

    # --- [A] PLAYBACK ---
    if not TRAIN_MODE:
        print("🎬 Starting Playback (Slanted Bird's Eye View)...")
        env = make_env_internal()
        h = ObsHistory(1, SEQ_LEN, obs_dim, device)
        o_r, _ = env.reset()
        h.prime(env._normalize_obs(o_r), 0)
        cum_r, n_ndr, st_t, s_t = 0.0, 0, 0, time.time()
        cam_d = False
        pbar = tqdm(desc="Playback", unit="step")
        try:
            while True:
                with torch.no_grad():
                    a_np = agent.get_action_and_value(h.get())[0].cpu() \
                        .numpy().flatten()
                on_r, rw, dn, tr, inf = env.step(a_np)
                h.update_batch(env._normalize_obs(on_r)[np.newaxis, :])
                st_t += 1
                cum_r += rw
                if not inf.get("is_detected", False):
                    n_ndr += 1
                pbar.update(1)
                pbar.set_postfix({
                    "SPS": int(st_t/(time.time()-s_t+1e-6)),
                    "Reward": f"{cum_r/st_t:.4f}",
                    "NDR": f"{n_ndr/st_t:.2f}"
                })
                draw_vision_lines(env)
                env.render()
                if not cam_d and env.viewer:
                    env.viewer.cam.lookat[:] = [0, 0, 0]
                    env.viewer.cam.distance = 15.0
                    env.viewer.cam.elevation = -50.0
                    cam_d = True
                time.sleep(0.04)
                if dn or tr:
                    o_res, _ = env.reset()
                    h.reset()
                    h.prime(env._normalize_obs(o_res), 0)
        except KeyboardInterrupt:
            pass
        finally:
            pbar.close()
            env.close()
            sys.exit(0)

    # --- [B] TRAINING ---
    else:
        print(f"🚀 Training: {NUM_ENVS} envs | Target: {TRAINING_TARGET}")
        envs_tr_list_v = [make_env_internal() for _ in range(NUM_ENVS)]
        optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
        writer = SummaryWriter(
            f"runs/main26_{TRAINING_TARGET}_{datetime.now().strftime('%m%d_%H%M')}"
        )

        obs_st = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, SEQ_LEN, obs_dim),
                             device=device)
        act_st = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, act_dim), device=device)
        lp_st = torch.zeros((ROLLOUT_STEPS, NUM_ENVS), device=device)
        rw_st = torch.zeros((ROLLOUT_STEPS, NUM_ENVS), device=device)
        dn_st = torch.zeros((ROLLOUT_STEPS, NUM_ENVS), device=device)
        vl_st = torch.zeros((ROLLOUT_STEPS, NUM_ENVS), device=device)

        h_tr = ObsHistory(NUM_ENVS, SEQ_LEN, obs_dim, device)
        ndr_q = deque(maxlen=NDR_WINDOW)
        for i in range(NUM_ENVS):
            oi, _ = envs_tr_list_v[i].reset()
            h_tr.prime(envs_tr_list_v[i]._normalize_obs(oi), i)

        n_dn, g_st, s_t_tr = torch.zeros(NUM_ENVS, device=device), 0, \
            time.time()
        for u_idx in range(1, TOTAL_STEPS // (NUM_ENVS * ROLLOUT_STEPS) + 1):
            if ANNEAL_LR:
                frac = 1.0 - (u_idx - 1.0) / (TOTAL_STEPS //
                                              (NUM_ENVS * ROLLOUT_STEPS))
                optimizer.param_groups[0]["lr"] = frac * LEARNING_RATE

            agent.eval()
            for ri in range(ROLLOUT_STEPS):
                g_st += NUM_ENVS
                c_seq = h_tr.get()
                obs_st[ri], dn_st[ri] = c_seq, n_dn
                with torch.no_grad():
                    at, lpt, _, vt = agent.get_action_and_value(c_seq)
                    vl_st[ri] = vt.flatten()
                act_st[ri], lp_st[ri] = at, lpt
                a_np = at.cpu().numpy()
                obs_raw_b = np.zeros((NUM_ENVS, obs_dim))
                for ie in range(NUM_ENVS):
                    on, r, d, tr, inf = envs_tr_list_v[ie].step(a_np[ie])
                    rw_st[ri, ie] = float(r)
                    if TRAINING_TARGET == "hider":
                        ndr_q.append(float(not inf.get("is_detected", False)))
                    is_f = bool(d or tr)
                    n_dn[ie] = float(is_f)
                    if is_f:
                        o_r, _ = envs_tr_list_v[ie].reset()
                        h_tr.update_single(envs_tr_list_v[ie].
                                           _normalize_obs(o_r), ie)
                        obs_raw_b[ie] = o_r
                    else:
                        obs_raw_b[ie] = on
                h_tr.update_batch(batch_normalize_obs(obs_raw_b))
                if USE_VIEWER:
                    draw_vision_lines(envs_tr_list_v[0])
                    envs_tr_list_v[0].render()

            with torch.no_grad():
                nv = agent.get_value(h_tr.get()).reshape(1, -1)
                advs = torch.zeros_like(rw_st, device=device)
                lg = 0
                for t in reversed(range(ROLLOUT_STEPS)):
                    va = nv if t == ROLLOUT_STEPS - 1 else vl_st[t + 1]
                    mask = 1.0 - dn_st[t]
                    delta = rw_st[t] + GAE_GAMMA * va * mask - vl_st[t]
                    lg = delta + GAE_GAMMA * GAE_LAMBDA * mask * lg
                    advs[t] = lg
                rets = advs + vl_st

            agent.train()
            b_obs = obs_st.reshape(-1, SEQ_LEN, obs_dim)
            b_lp = lp_st.reshape(-1)
            b_act = act_st.reshape(-1, act_dim)
            b_adv = advs.reshape(-1)
            b_ret = rets.reshape(-1)
            b_vl = vl_st.reshape(-1)

            # 詳細統計アキュムレータ初期化
            apg, avl, aet, akl, btc = 0.0, 0.0, 0.0, 0.0, 0
            indices = np.arange(NUM_ENVS * ROLLOUT_STEPS)

            for ep in range(UPDATE_EPOCHS):
                np.random.shuffle(indices)
                m_sz = (NUM_ENVS * ROLLOUT_STEPS) // NUM_MINIBATCHES
                for si in range(0, NUM_ENVS * ROLLOUT_STEPS, m_sz):
                    mb = indices[si:si+m_sz]
                    _, nlp, net, nv = agent.get_action_and_value(b_obs[mb],
                                                                b_act[mb])
                    ratio = (nlp - b_lp[mb]).exp()

                    with torch.no_grad():
                        akl += (ratio - 1.0 - (nlp - b_lp[mb])).mean().item()

                    m_adv = b_adv[mb]
                    if NORM_ADV:
                        m_adv = (m_adv - m_adv.mean()) / (m_adv.std() + 1e-8)

                    pg1, pg2 = -m_adv * ratio, -m_adv * torch.clamp(
                        ratio, 1-CLIP_COEF, 1+CLIP_COEF
                    )
                    pg_l = torch.max(pg1, pg2).mean()

                    nvf = nv.flatten()
                    if CLIP_VALUE_LOSS:
                        v_u = (nvf - b_ret[mb])**2
                        v_c = b_vl[mb] + torch.clamp(
                            nvf - b_vl[mb], -CLIP_COEF, CLIP_COEF
                        )
                        v_l = 0.5 * torch.max(v_u, (v_c - b_ret[mb])**2).mean()
                    else:
                        v_l = 0.5 * ((nvf - b_ret[mb])**2).mean()

                    total_l = pg_l - ENTROPY_COEF * net.mean() + v_l * VF_COEF
                    apg += pg_l.item()
                    avl += v_l.item()
                    aet += net.mean().item()
                    btc += 1

                    optimizer.zero_grad()
                    total_l.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                if TARGET_KL is not None and (akl / btc) > TARGET_KL:
                    break

            if u_idx % LOG_FREQ == 0:
                elapsed = time.time() - start_time_tr_v
                sps = int(g_st / (elapsed + 1e-6))
                ndr = np.mean(ndr_q) if ndr_q else 0.0
                div = btc + 1e-6
                m_rew, m_pg, m_vl, m_et, m_kl = rw_st.mean().item(), apg/div, \
                    avl/div, aet/div, akl/div
                print(f"Update {u_idx:4d} | Step {g_st:8d} | SPS {sps:5d} | "
                      f"Rew {m_rew:7.4f} | NDR {ndr:4.2f} | P-L {m_pg:6.4f} | "
                      f"V-L {m_vl:6.4f} | Ent {m_et:6.4f}")
                if TRACK_WANDB:
                    log = {
                        "charts/SPS": sps,
                        "charts/reward": m_rew,
                        "charts/ndr": ndr,
                        "losses/policy_loss": m_pg,
                        "losses/value_loss": m_vl,
                        "losses/entropy": m_et,
                        "losses/approx_kl": m_kl,
                        "charts/learning_rate": optimizer.param_groups[0]["lr"]
                    }
                    wandb.log(log, step=g_st)
            if u_idx % SAVE_FREQ == 0:
                torch.save(agent.state_dict(), m_path)

        for env_f in envs_tr_list_v:
            env_f.close()
        if TRACK_WANDB:
            wandb.finish()


if __name__ == "__main__":
    run()