#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.getcwd())

import main18_optimization as base_config
import mujoco

xml_string = base_config.XML_CONTENT
model = mujoco.MjModel.from_xml_string(xml_string)

# Agent geoms を確認
agent_geoms = ['seeker_geom', 'hider1_geom', 'hider2_geom']
for g_name in agent_geoms:
    try:
        geom_id = model.geom(g_name).id
        geom_size = model.geom_size[geom_id]
        print(f"{g_name}: size={geom_size}")
        # 球体の場合、サイズ[0]が半径
        print(f"  -> radius={geom_size[0]}, diameter={2*geom_size[0]}")
    except:
        print(f"{g_name}: NOT FOUND")
