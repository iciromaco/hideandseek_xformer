#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.getcwd())

import main18_optimization as base_config
import mujoco

xml_string = base_config.XML_CONTENT
model = mujoco.MjModel.from_xml_string(xml_string)

bodies = ['seeker_body', 'hider1_body', 'hider2_body', 'box1_body', 'box2_body', 'ramp_body']
for b in bodies:
    try:
        body_id = model.body(b).id
        print(f"{b}: id={body_id}")
    except:
        print(f"{b}: NOT FOUND")
