# Smoke-test TeamCosEnv
import sys, os
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.envs.hns_environment import TeamCosEnv
print('Creating env...')
try:
    env = TeamCosEnv(mode='initial', target='seeker', n_seekers=1, n_hiders=2, n_boxes=0, n_ramps=0, render_mode=None)
    print('Resetting...')
    obs, info = env.reset()
    print('Reset OK, info=', info)
    for i in range(8):
        obs, r, _, done, info = env.step([0.0, 0.0, 0.0, 0.0])
        print(f'FRAME {i}: reward={r:.4f} is_detected={info.get("is_detected")}')
    env.close()
except Exception as e:
    print('ERROR during smoke test:', type(e), e)
    import traceback; traceback.print_exc()
    sys.exit(1)
print('Smoke test finished successfully')
