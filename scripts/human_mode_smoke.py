import importlib.util
import sys
import time
from pathlib import Path

module_path = Path(__file__).resolve().parents[0] / '..' / 'main28_train_final.py'
module_path = module_path.resolve()
source = module_path.read_text()

import ast
tree = ast.parse(source)

def extract_func_src(name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    return None

src_try = extract_func_src('_try_call_env_method')
src_human = extract_func_src('run_human_mode')
if src_try is None or src_human is None:
    print('Could not extract functions from source')
    raise SystemExit(2)

# Prepare execution namespace with minimal fakes
ns = {}

class FakeNP:
    float32 = float
    generic = ()
    class FakeArray(list):
        def __init__(self, data):
            super().__init__(data)
            # naive shape inference
            self.shape = (len(data),)
            self.size = len(data)
            self.ndim = 1

    def zeros(self, shape, dtype=None):
        if isinstance(shape, tuple):
            n = 1
            for s in shape:
                n *= s
            return FakeNP.FakeArray([0.0] * n)
        return FakeNP.FakeArray([0.0] * shape)
    def asarray(self, v, dtype=None):
        return FakeNP.FakeArray(list(v))
    def array(self, v, dtype=None):
        return FakeNP.FakeArray(list(v))
    def mean(self, v):
        if not v:
            return 0.0
        return sum(v) / len(v)

ns['np'] = FakeNP()
ns['time'] = time

class _FakeListener:
    def __init__(self, on_press=None, on_release=None):
        self.on_press = on_press
        self.on_release = on_release
    def start(self):
        return None
    def stop(self):
        return None

class _FakePynput:
    Listener = _FakeListener

ns['_pynput_keyboard'] = _FakePynput()

class DummyActionSpace:
    def __init__(self):
        self.shape = (2,)

class DummyEnv:
    def __init__(self):
        self.action_space = DummyActionSpace()
    def reset(self):
        obs = [0.0] * 10
        return obs, {}
    def step(self, action):
        obs = [0.0] * 10
        return obs, 0.0, False, False, {}
    def render(self):
        return None
    def close(self, terminate=True):
        return None
    def grab(self):
        return None
    def lock(self):
        return None

ns['build_env'] = lambda mode, rt, cfg, render_mode=None: DummyEnv()
ns['ENV_CONFIG'] = {}
ns['MODE'] = 'refinement'
ns['_resolve_runtime_target'] = lambda : 'seeker'

# Execute extracted functions into namespace
exec(src_try, ns)
exec(src_human, ns)

print('Starting smoke run of extracted run_human_mode (will exit via timeout)...')
try:
    rc = ns['run_human_mode']('seeker')
    print('run_human_mode returned', rc)
except Exception as exc:
    print('run_human_mode raised', repr(exc))

print('Smoke run finished')
