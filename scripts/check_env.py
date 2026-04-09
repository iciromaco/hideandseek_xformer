import sys
sys.path.insert(0, '.')
from src.envs.hns_environment import TeamCosEnv

print('Creating TeamCosEnv (render_mode=None) ...')
env = TeamCosEnv(render_mode=None)
print('Created TeamCosEnv successfully')
print('Model body count (nbody):', getattr(env.model, 'nbody', 'unknown'))

# Prefer name-based lookups to numeric indexing (safer for XML changes)
expected = ['floor']
# derive agent body names from the environment's agent_keys
try:
	for ak in getattr(env, 'agent_keys', []):
		expected.append(f"{ak}_body")
except Exception:
	pass

# derive object body names from counts
try:
	for i in range(1, int(getattr(env, 'n_boxes', 0)) + 1):
		expected.append(f"box{i}_body")
	for i in range(1, int(getattr(env, 'n_ramps', 0)) + 1):
		expected.append(f"ramp{i}_body")
except Exception:
	pass

for name in expected:
	try:
		bid = env.model.body(name).id
		print(f"Found body '{name}' -> id={bid}")
	except Exception as e:
		print(f"Body '{name}' not found or access failed: {e}")

# As a fallback, print a small slice of a stable numeric array (body_mass)
try:
	bm = env.model.body_mass
	print('len(body_mass):', len(bm))
	print('first body masses:', list(bm[:10]))
except Exception:
	pass

print('Done')
