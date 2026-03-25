from src.envs.hns_environment import TeamCosEnv

print("Creating env...")
env = TeamCosEnv(debug_mode=True, debug_console=True, mode="initial", render_mode=None)
print("Resetting env...")
try:
    r = env.reset()
    print("reset ok ->", type(r))
except Exception as e:
    print("reset failed:", e)

key = env.learnable_agent_key
print("learnable key:", key)
env.last_debug_ctrl[key] = (1.0, 0.0)
action = [1.0, 0.0, 0.0, 0.0]
print("Running steps...")
for i in range(120):
    try:
        _ = env.step(action)
        if i % 10 == 0:
            print(f"step {i} ok")
    except Exception as e:
        print("step error:", e)
        break
print("DONE")
