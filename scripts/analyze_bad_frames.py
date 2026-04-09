import csv
import ast
import math
from statistics import mean

fn = 'bad_frames_obs_summary.csv'

rows = []
with open(fn, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        # parse numeric fields
        r['det_action0'] = float(r['det_action0'])
        r['wall_distance'] = float(r['wall_distance']) if r['wall_distance']!='' else math.nan
        r['step'] = int(r['step'])
        # parse obs
        try:
            r['obs'] = ast.literal_eval(r['obs'])
        except Exception:
            r['obs'] = []
        rows.append(r)

def lidar_front_min(obs):
    # LIDAR is slice 5:17; front indices within lidar: [0,1,2]
    base = 5
    idxs = [base + i for i in (0,1,2)]
    vals = [obs[i] for i in idxs if i < len(obs)]
    return min(vals) if vals else math.nan

def lidar_back(obs):
    base = 5
    back_idx = base + 11
    return obs[back_idx] if back_idx < len(obs) else math.nan

front_mins = [lidar_front_min(r['obs']) for r in rows]
backs = [lidar_back(r['obs']) for r in rows]
det_actions = [r['det_action0'] for r in rows]
wall_ds = [r['wall_distance'] for r in rows]

print(f"Rows: {len(rows)}")
print(f"det_action0 mean: {mean(det_actions):.4f}")
print(f"wall_distance mean: {mean(wall_ds):.4f}")
print(f"front_min mean: {mean(front_mins):.4f}")
print(f"back mean: {mean(backs):.4f}")

# correlation (pearson) det_action0 vs front_min
def pearson(x,y):
    n = len(x)
    xm = mean(x); ym = mean(y)
    num = sum((a-xm)*(b-ym) for a,b in zip(x,y))
    den = math.sqrt(sum((a-xm)**2 for a in x)*sum((b-ym)**2 for b in y))
    return num/den if den>0 else math.nan

print(f"pearson(det_action0, front_min): {pearson(det_actions, front_mins):.4f}")

# find frames where det_action0 < -0.2 but front_min > 0.3 (backing despite clear front)
cases = [r for r,f in zip(rows, front_mins) if r['det_action0'] < -0.2 and f > 0.3]
print(f"cases det_action0<-0.2 & front_min>0.3: {len(cases)}")
for c in cases[:10]:
    step = c['step']
    print(f"step {step}: det_action0={c['det_action0']:.3f}, wall_distance={c['wall_distance']:.3f}, front_min={lidar_front_min(c['obs']):.3f}")

with open('analysis_bad_frames.txt','w') as out:
    out.write('Summary of bad frames analysis\n')
    out.write(f"Rows: {len(rows)}\n")
    out.write(f"det_action0 mean: {mean(det_actions):.4f}\n")
    out.write(f"wall_distance mean: {mean(wall_ds):.4f}\n")
    out.write(f"front_min mean: {mean(front_mins):.4f}\n")
    out.write(f"back mean: {mean(backs):.4f}\n")
    out.write(f"pearson(det_action0, front_min): {pearson(det_actions, front_mins):.4f}\n")
    out.write(f"cases det_action0<-0.2 & front_min>0.3: {len(cases)}\n")
    for c in cases[:50]:
        out.write(f"step {c['step']}, det_action0={c['det_action0']}, wall_distance={c['wall_distance']}, front_min={lidar_front_min(c['obs'])}\n")

print('Wrote analysis_bad_frames.txt')
