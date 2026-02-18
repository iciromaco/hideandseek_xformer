#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.getcwd())

import main18_optimization as base_config
import mujoco
import numpy as np

xml_string = base_config.XML_CONTENT
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# ramp_body の位置を確認
ramp_body_id = model.body("ramp_body").id
ramp_pos = data.xpos[ramp_body_id][:2]
print(f"Ramp position in MuJoCo: {ramp_pos}")
print(f"Ramp full xpos: {data.xpos[ramp_body_id]}")

# ramp_geom の情報
try:
    ramp_geom_id = model.geom("ramp_geom").id
    print(f"Ramp geom size: {model.geom_size[ramp_geom_id]}")
except:
    print("Ramp geom not found by name 'ramp_geom'")
    # すべてのgeomから ramp を含むものを探す
    for i in range(model.ngeom):
        geom = model.geom(i)
        if "ramp" in geom.name:
            print(f"Found: {geom.name}, size={model.geom_size[i]}")
