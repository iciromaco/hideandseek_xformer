import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.hns_environment import TeamCosEnv


def main():
    env = TeamCosEnv(debug_mode=False)
    learnable = env.learnable_agent_key
    body_id = env.body_ids[learnable]
    print('learnable agent key:', learnable)
    print('body id:', body_id)
    b_jntadr = env.model.body_jntadr[body_id]
    print('body_jntadr:', b_jntadr)
    # list joint ids starting at body_jntadr
    for j in range(max(0, b_jntadr - 2), b_jntadr + 6):
        try:
            jname = env.model.joint(j).name
        except Exception:
            jname = '<no-joint>'
        j_dofadr = env.model.jnt_dofadr[j] if j < len(env.model.jnt_dofadr) else None
        print('joint', j, 'name', jname, 'jnt_dofadr', j_dofadr)

    print('\nactuator mapping (name -> id):')
    for name, aid in env.actuator_ids.items():
        print(name, '->', aid)

    # show first 40 qvel entries
    qvel = env.data.qvel
    print('\nqvel shape:', qvel.shape)
    print('qvel[0:40]:', list(map(float, qvel[:40])))


if __name__ == '__main__':
    main()
