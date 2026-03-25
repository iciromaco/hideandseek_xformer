"""Generate and save mode4 SDF grid by instantiating the environment.
Run:
    python scripts/generate_sdf.py
This will create the sdfgrid_mode4_*.pkl under the `envs/` folder.
"""

import glob
import os
import time

from envs.hns_environment import TeamCosEnv


def find_latest_sdf(envs_dir):
    pattern = os.path.join(envs_dir, "sdfgrid_mode4_*.pkl")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


def main():
    print("Instantiating environment to generate SDF (this may take a few seconds)...")
    env = TeamCosEnv(mode="initial", target="seeker", n_seekers=1, n_hiders=1, n_boxes=0, n_ramps=0, render_mode=None, debug_mode=False)
    try:
        env.vis_engine._ensure_mode4_sdf()
        env.close()
    except Exception as e:
        try:
            env.close()
        except Exception:
            pass
        raise

    # visibility_engine saves into src/envs; check both locations
    repo_envs = os.path.join(os.path.dirname(__file__), "../envs")
    src_envs = os.path.join(os.path.dirname(__file__), "../src/envs")
    latest = find_latest_sdf(src_envs) or find_latest_sdf(repo_envs)
    if latest is None:
        print(f"No sdf file found in {envs_dir}")
    else:
        mtime = time.ctime(os.path.getmtime(latest))
        print(f"Generated SDF file: {latest}\n  modified: {mtime}")


if __name__ == "__main__":
    main()
