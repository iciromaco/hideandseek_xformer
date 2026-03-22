import pathlib
import sys
import traceback

p = pathlib.Path.cwd()
print("CWD:", p)
sys.path.insert(0, str(p))
try:
    import importlib.util

    spec = importlib.util.spec_from_file_location("hns_env", "src/envs/hns_environment.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    print("IMPORT_OK")
except Exception:
    print("IMPORT_FAILED")
    traceback.print_exc()
    raise
