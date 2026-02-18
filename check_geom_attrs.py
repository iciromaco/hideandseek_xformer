#!/usr/bin/env python
import mujoco
import main18_optimization as base_config

xml_string = base_config.XML_CONTENT
model = mujoco.MjModel.from_xml_string(xml_string)

print("[BOX GEOMETRIES]")
for gid in range(model.ngeom):
    g = model.geom(gid)
    body_name = model.body(int(g.bodyid)).name
    if "box" in g.name or "box" in body_name:
        print(f"Geom {gid}: {g.name}")
        print(f"  Body: {body_name}")
        print(f"  contype: {g.contype}")
        print(f"  conaffinity: {g.conaffinity}")
        print()

print("[WALL GEOMETRIES (first few)]")
count = 0
for gid in range(model.ngeom):
    g = model.geom(gid)
    body = model.body(int(g.bodyid)).name
    if body == "world" and ("wall" in g.name or "maze" in g.name):
        print(f"Geom {gid}: {g.name}")
        print(f"  contype: {g.contype}")
        print(f"  conaffinity: {g.conaffinity}")
        count += 1
        if count >= 2:
            break
