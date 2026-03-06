#!/usr/bin/env python3
"""
static_walls が正しく抽出されているか確認
"""

import sys
sys.path.insert(0, "/Users/dan/Desktop/Semi/hideandseek_xformer")

from main23_sightmap_optimized import StaticWallManager

# StaticWallManager を初期化（XMLを自動読み込み）
try:
    manager = StaticWallManager(use_xml=True)
    print(f"Static walls count: {len(manager.static_walls)}")
    print()
    
    if len(manager.static_walls) > 0:
        print("[STATIC WALLS]")
        for i, wall in enumerate(manager.static_walls):
            start, end = wall
            print(f"Wall {i}: {start} -> {end}")
    else:
        print("ERROR: No static walls loaded!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
