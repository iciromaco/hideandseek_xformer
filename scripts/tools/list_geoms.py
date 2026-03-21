#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.getcwd())

import main18_optimization as base_config
import mujoco

xml_string = base_config.XML_CONTENT
model = mujoco.MjModel.from_xml_string(xml_string)

# すべてのgeomを列挙
print("All geoms:")
for i in range(model.ngeom):
    geom = model.geom(i)
    geom_name = geom.name
    body_id = geom.bodyid
    body_name = model.body(int(body_id)).name if body_id >= 0 else "none"
    geom_size = model.geom_size[i]
    print(f"  Geom {i}: {geom_name} (body: {body_name}), size={geom_size}")
