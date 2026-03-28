import json
import os

p = "experiments/contact_trace.json"
if not os.path.exists(p):
    print("missing", p)
    raise SystemExit(1)
with open(p, "r") as f:
    d = json.load(f)
recs = d.get("records", [])
for r in recs:
    if r.get("step") == 288:
        import pprint

        pprint.pprint(r)
        break
else:
    print("no step 288")
