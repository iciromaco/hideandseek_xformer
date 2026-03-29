import math
import os

import numpy as np
from PIL import Image

from src.envs.hns_environment import TeamCosEnv


def pixel_to_world(x, y, img_w, img_h, arena_half=6.0):
    # map pixel coords (origin top-left) to world coords centered at (0,0)
    wx = (float(x) / float(img_w)) * 2.0 * arena_half - arena_half
    wy = arena_half - (float(y) / float(img_h)) * 2.0 * arena_half
    return wx, wy


def detect_regions(img_path):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    yellow_mask = (r > 150) & (g > 140) & (b < 120) & (r > g - 50)
    blue_mask = (b > 140) & (r < 120) & (g < 120)

    def center_from_mask(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return int(xs.mean()), int(ys.mean())

    y_c = center_from_mask(yellow_mask)
    b_c = center_from_mask(blue_mask)
    return (w, h), y_c, b_c


def main(img_file="スクリーンショット.png"):
    cwd = os.getcwd()
    p = os.path.join(cwd, img_file)
    if not os.path.exists(p):
        print("image not found:", p)
        return

    (w, h), yellow_c, blue_c = detect_regions(p)
    print("img size", w, h)
    print("yellow center", yellow_c)
    print("blue center", blue_c)

    env = TeamCosEnv(render_mode=None)
    obs, info = env.reset()
    print("agent keys:", env.agent_keys)
    # choose first seeker and first hider
    seekers = [k for k in env.agent_keys if k.startswith("s")]
    hiders = [k for k in env.agent_keys if k.startswith("h")]
    if not seekers or not hiders:
        print("no agents found")
        return
    sk = seekers[0]
    hk = hiders[0]
    print("testing pair", sk, hk)

    # map pixel centers to world coordinates
    if yellow_c is None or blue_c is None:
        print("could not detect colored regions in image")
        return
    yellow_w = pixel_to_world(yellow_c[0], yellow_c[1], w, h, arena_half=env.ARENA_HALF)
    blue_w = pixel_to_world(blue_c[0], blue_c[1], w, h, arena_half=env.ARENA_HALF)
    print("mapped yellow->world", yellow_w)
    print("mapped blue->world", blue_w)

    # place seeker at blue location, hider at yellow location; set z to 0.4
    s_bid = env.body_ids[sk]
    h_bid = env.body_ids[hk]
    env.data.xpos[s_bid][0] = float(blue_w[0])
    env.data.xpos[s_bid][1] = float(blue_w[1])
    env.data.xpos[s_bid][2] = 0.4
    env.data.xpos[h_bid][0] = float(yellow_w[0])
    env.data.xpos[h_bid][1] = float(yellow_w[1])
    env.data.xpos[h_bid][2] = 0.4

    # set seeker rotation to face from seeker->hider
    dx = yellow_w[0] - blue_w[0]
    dy = yellow_w[1] - blue_w[1]
    rot = math.atan2(dy, dx)
    ridx = env.model.jnt_qposadr[env.qpos_indices[sk]["rot"]]
    env.data.qpos[ridx] = float(rot)

    # call visibility engine
    p1 = np.array([env.data.xpos[s_bid][0], env.data.xpos[s_bid][1], 0.4])
    p2 = np.array([env.data.xpos[h_bid][0], env.data.xpos[h_bid][1], 0.4])
    try:
        vis = bool(env.vis_engine.is_visible(p1, p2, body_exclude=s_bid, target_body_id=h_bid))
    except Exception as e:
        print("vis_engine raised:", e)
        vis = False

    print(f"VisibilityEngine.is_visible({sk}->{hk}) = {vis}")
    # also print cached wrapper result
    print("_is_vis wrapper:", env._is_vis(env.data.xpos[s_bid][:2], float(env.data.qpos[ridx]), env.data.xpos[h_bid][:2], s_bid, h_bid))


if __name__ == "__main__":
    main()
