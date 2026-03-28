#!/usr/bin/env python3
"""SPS測定用スクリプト（4つの最適化適用版）"""

import time

from main23_sightmap_optimized import TeamCosEnv


def test_sps():
    """SPS測定"""
    env = TeamCosEnv()
    obs, _ = env.reset()

    print("Warming up...")
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    print("\nMeasuring SPS...")
    n_steps = 1000
    t_start = time.time()

    for step_idx in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

        if (step_idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            sps = (step_idx + 1) / elapsed
            print(f"Step {step_idx + 1}: {sps:.0f} SPS")

    elapsed = time.time() - t_start
    sps = n_steps / elapsed
    print(f"\n✓ Final SPS: {sps:.0f} (steps/sec)")
    print(f"  Total time: {elapsed:.2f} sec for {n_steps} steps")


if __name__ == "__main__":
    try:
        test_sps()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
