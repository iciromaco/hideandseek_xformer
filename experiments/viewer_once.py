import os
import sys
import subprocess
import time
import mujoco

def move_boxes_ramps_far(env, far_offset=20.0):
    # boxes
    if getattr(env, 'box_ids', None):
        for i, bid in enumerate(env.box_ids):
            x = far_offset + 2.0 * i
            y = far_offset
            adr = env.model.jnt_qposadr[env.model.body_jntadr[bid]]
            env.data.qpos[adr:adr+7] = [float(x), float(y), 0.5, 1.0, 0.0, 0.0, 0.0]
    # ramps
    if getattr(env, 'ramp_ids', None):
        for i, rid in enumerate(env.ramp_ids):
            x = far_offset
            y = far_offset + 4.0 + 2.0 * i
            adr = env.model.jnt_qposadr[env.model.body_jntadr[rid]]
            env.data.qpos[adr:adr+7] = [float(x), float(y), 0.0, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)



ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'src'))


CHILD_FLAG = '--_viewer_child'


def move_boxes_ramps_far(env, far_offset=20.0):
    # boxes
    if getattr(env, 'box_ids', None):
        for i, bid in enumerate(env.box_ids):
            x = far_offset + 2.0 * i
            y = far_offset
            adr = env.model.jnt_qposadr[env.model.body_jntadr[bid]]
            env.data.qpos[adr:adr+7] = [float(x), float(y), 0.5, 1.0, 0.0, 0.0, 0.0]
    # ramps
    if getattr(env, 'ramp_ids', None):
        for i, rid in enumerate(env.ramp_ids):
            x = far_offset
            y = far_offset + 4.0 + 2.0 * i
            adr = env.model.jnt_qposadr[env.model.body_jntadr[rid]]
            env.data.qpos[adr:adr+7] = [float(x), float(y), 0.0, 1.0, 0.0, 0.0, 0.0]
    import mujoco
    mujoco.mj_forward(env.model, env.data)


def run_child():

    # Child creates the env, places hider, and launches blocking viewer
    try:
        from envs.hns_environment import TeamCosEnv
        from experiments.online_update_untrained import set_hider_pos_fixed
        from experiments.utils import prepare_env
        import mujoco
    except Exception as e:
        print('Child import error:', e)
        raise

    env = TeamCosEnv(debug_mode=True, target='seeker')
    # Hider(h1)をother_agentsから除外してprepare_envを呼ぶ
    h_key = env.hider_keys[0]
    # agent_keysからh_keyを除外したリストを作成
    env.agent_keys = [k for k in env.agent_keys if k != h_key]
    env = prepare_env(env, action_repeat=None, place_far=True)

    obs, info = env.reset()
    # reset後に箱・ランプを再度遠方に飛ばす
    move_boxes_ramps_far(env, far_offset=20.0)
    # Seekerの位置・向き取得
    s_key = env.learnable_agent_key
    s_bid = env.body_ids[s_key]
    s_pos = env.data.xpos[s_bid][:2].copy()
    try:
        rot_jid = env.qpos_indices[s_key]['rot']
        qadr = env.model.jnt_qposadr[rot_jid]
        yaw = float(env.data.qpos[qadr])
    except Exception:
        try:
            q = env.data.xquat[s_bid]
            q0, q1, q2, q3 = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            import numpy as np
            yaw = float(np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3)))
        except Exception:
            yaw = 0.0
    # HiderをSeeker前方に固定
    hx = float(s_pos[0] + 1.5 * __import__('math').cos(yaw))
    hy = float(s_pos[1] + 1.5 * __import__('math').sin(yaw))
    set_hider_pos_fixed(env, h_key, hx, hy, z=0.8)
    try:
        mujoco.mj_forward(env.model, env.data)
    except Exception:
        pass
    print(f"[INFO] 箱・ランプ・他エージェントは遠方、Hider({h_key})のみSeeker前方に配置しました。Enterで終了します。")

    print('Child: launching mujoco.viewer (close window to exit, or press Enter in parent)')
    try:
        mujoco.viewer.launch(env.model, env.data)
    except Exception as e:
        print('Child viewer.launch error:', e)
        raise
    import sys
    sys.exit(0)


if __name__ == '__main__':
    if CHILD_FLAG in sys.argv:
        run_child()
        sys.exit(0)

    # Parent: spawn child as same interpreter, then wait for Enter to kill
    py = sys.executable
    cmd = [py, os.path.abspath(__file__), CHILD_FLAG]
    env = os.environ.copy()
    print('Parent: launching viewer child with', cmd)
    proc = subprocess.Popen(cmd, env=env)
    try:
        input('Viewer launched (pid=%d). Press Enter to terminate viewer and exit.\n' % proc.pid)
    except KeyboardInterrupt:
        pass
    print('Parent: terminating child...')
    import signal
    try:
        proc.terminate()
        time.sleep(0.2)
        if proc.poll() is None:
            proc.kill()
        time.sleep(0.2)
        if proc.poll() is None:
            print('Parent: sending SIGKILL to child...')
            os.kill(proc.pid, signal.SIGKILL)
    except Exception as e:
        print(f'Parent: failed to kill child: {e}')
    print('Parent: done')
