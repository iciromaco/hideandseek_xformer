import sys,os,time
# sys.path拡張：このスクリプトの親ディレクトリ（プロジェクトルート）と src を追加
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from envs.hns_environment import TeamCosEnv
from experiments.utils import prepare_env

env = TeamCosEnv(debug_mode=False, target="seeker")
# env.reset()
# move boxes/ramps/other agents far after the initial reset so reset()'s random placement
# is not overwritten by a later reset
env = prepare_env(env, place_far=True)
a = [0.25,0.0,0.0,0.0]
for _ in range(400):
    env.step(a)
    env.render()
    time.sleep(0.05)  # 50 ms 暫定
input("Viewer open — press Enter to close")
env.close()