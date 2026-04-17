#!/usr/bin/env python3
"""
xml_refinement_tool.py test suite
Validates XML generation, model loading, and control
"""

import sys
from xml_refinement_tool import XMLRefinementApp, MinimalEnvConfig, XMLBuilder
import mujoco


def test_xml_generation():
    """Test XML generation"""
    print("=" * 60)
    print("Test 1: XML Generation")
    print("=" * 60)
    
    config = MinimalEnvConfig(n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1)
    builder = XMLBuilder(config)
    xml_string = builder.build_xml()
    
    checks = {
        "<mujoco>": "root XML",
        "<worldbody>": "world body",
        'name="floor"': "floor",
        'name="wall_n"': "north wall",
        'name="wall_s"': "south wall",
        'name="wall_e"': "east wall",
        'name="wall_w"': "west wall",
        "ramp1_body": "ramp",
        "box1_body": "box",
        "s_anchor": "seeker agent",
        "h1_anchor": "hider agent",
        "<actuator>": "actuators",
        "<equality>": "equality constraints",
    }
    
    all_ok = True
    for check, desc in checks.items():
        if check in xml_string:
            print(f"  [OK] {desc}")
        else:
            print(f"  [FAIL] {desc}")
            all_ok = False
    
    print()
    return all_ok


def test_model_loading():
    """Test MuJoCo model loading"""
    print("=" * 60)
    print("Test 2: MuJoCo Model Loading")
    print("=" * 60)
    
    try:
        app = XMLRefinementApp(n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1, interactive=False)
        model = app.model
        data = app.data
        
        print(f"  [OK] Model loaded successfully")
        print(f"    - Bodies: {model.nbody}")
        print(f"    - Joints: {model.njnt}")
        print(f"    - Actuators: {model.nu}")
        print(f"    - Geoms: {model.ngeom}")
        print(f"    - Equality constraints: {model.neq}")
        print()
        return True
    except Exception as e:
        print(f"  [FAIL] Model loading error: {e}")
        print()
        return False


def test_agent_control():
    """Test agent control"""
    print("=" * 60)
    print("Test 3: Agent Control")
    print("=" * 60)
    
    try:
        app = XMLRefinementApp(n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1, interactive=False)
        model = app.model
        data = app.data
        
        try:
            fwd_act_id = model.actuator("s_fwd").id
            turn_act_id = model.actuator("s_turn").id
            
            print(f"  [OK] Seeker actuators found")
            print(f"    - Forward actuator ID: {fwd_act_id}")
            print(f"    - Turn actuator ID: {turn_act_id}")
            
            for step in range(10):
                data.ctrl[fwd_act_id] = 0.5
                data.ctrl[turn_act_id] = 0.3
                mujoco.mj_step(model, data)
            
            print(f"  [OK] Simulation 10 steps successful")
            print(f"    - Current time: {data.time:.3f}s")
            print()
            return True
        except Exception as e:
            print(f"  [FAIL] Actuator operation error: {e}")
            print()
            return False
    except Exception as e:
        print(f"  [FAIL] Control test error: {e}")
        print()
        return False


def test_parameter_customization():
    """Test parameter customization"""
    print("=" * 60)
    print("Test 4: Parameter Customization")
    print("=" * 60)
    
    try:
        class CustomConfig(MinimalEnvConfig):
            AGENT_MASS = 2.0
            BOX_MASS = 1.0
            RAMP_MASS = 2.0
            AGENT_DAMPING_XY = 0.5
        
        config = CustomConfig()
        builder = XMLBuilder(config)
        xml_string = builder.build_xml()
        
        model = mujoco.MjModel.from_xml_string(xml_string)
        
        print(f"  [OK] Custom config model generated")
        print(f"    - AGENT_MASS: {config.AGENT_MASS}")
        print(f"    - BOX_MASS: {config.BOX_MASS}")
        print(f"    - RAMP_MASS: {config.RAMP_MASS}")
        print(f"    - AGENT_DAMPING_XY: {config.AGENT_DAMPING_XY}")
        print()
        return True
    except Exception as e:
        print(f"  [FAIL] Customization test error: {e}")
        print()
        return False


def test_different_configurations():
    """Test different environment configurations"""
    print("=" * 60)
    print("Test 5: Different Environment Configs")
    print("=" * 60)
    
    configs = [
        {"n_seekers": 1, "n_hiders": 1, "n_boxes": 1, "n_ramps": 0},
        {"n_seekers": 2, "n_hiders": 3, "n_boxes": 4, "n_ramps": 2},
        {"n_seekers": 1, "n_hiders": 2, "n_boxes": 0, "n_ramps": 1},
    ]
    
    all_ok = True
    for i, cfg in enumerate(configs, 1):
        try:
            app = XMLRefinementApp(
                n_seekers=cfg["n_seekers"],
                n_hiders=cfg["n_hiders"],
                n_boxes=cfg["n_boxes"],
                n_ramps=cfg["n_ramps"],
                interactive=False
            )
            print(f"  [OK] Config {i}: S={cfg['n_seekers']}, H={cfg['n_hiders']}, "
                  f"B={cfg['n_boxes']}, R={cfg['n_ramps']}")
        except Exception as e:
            print(f"  [FAIL] Config {i} error: {e}")
            all_ok = False
    
    print()
    return all_ok


def main():
    """Run all tests"""
    print("\n")
    print("=" * 60)
    print("XML Refinement Tool - Test Suite")
    print("=" * 60)
    print("\n")
    
    results = []
    
    try:
        results.append(("XML generation", test_xml_generation()))
    except Exception as e:
        print(f"ERROR in test_xml_generation: {e}\n")
        results.append(("XML generation", False))
    
    try:
        results.append(("Model loading", test_model_loading()))
    except Exception as e:
        print(f"ERROR in test_model_loading: {e}\n")
        results.append(("Model loading", False))
    
    try:
        results.append(("Agent control", test_agent_control()))
    except Exception as e:
        print(f"ERROR in test_agent_control: {e}\n")
        results.append(("Agent control", False))
    
    try:
        results.append(("Parameter customization", test_parameter_customization()))
    except Exception as e:
        print(f"ERROR in test_parameter_customization: {e}\n")
        results.append(("Parameter customization", False))
    
    try:
        results.append(("Different configs", test_different_configurations()))
    except Exception as e:
        print(f"ERROR in test_different_configurations: {e}\n")
        results.append(("Different configs", False))
    
    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")
    
    print()
    print(f"Total: {passed}/{total} passed")
    print()
    
    if passed == total:
        print("= All tests passed!")
        return 0
    else:
        print(f"= {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
