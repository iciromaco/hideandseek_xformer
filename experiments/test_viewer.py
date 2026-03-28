import importlib
import os
import time
import traceback

xml = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="box" pos="0 0 0.5">
      <geom type="box" size="0.2 0.2 0.2"/>
    </body>
  </worldbody>
</mujoco>
"""

OUT_PNG = "experiments/test_viewer_frame.png"


def try_backends():
    # Try a set of likely backends by setting MUJOCO_GL before reloading context
    # Prefer the currently-set MUJOCO_GL, then plausible fallbacks (include macOS 'cgl')
    env_pref = os.environ.get("MUJOCO_GL")
    fallbacks = ["cgl", "glfw", "osmesa", "egl", "gl"]
    candidates = []
    if env_pref:
        candidates.append(env_pref)
    for b in fallbacks:
        if b not in candidates:
            candidates.append(b)
    for b in candidates:
        print("---")
        print("Trying MUJOCO_GL=", b)
        os.environ["MUJOCO_GL"] = b
        try:
            import mujoco

            importlib.reload(mujoco.gl_context)
            print(
                "mujoco.gl_context keys=",
                [k for k in dir(mujoco.gl_context) if not k.startswith("__")],
            )
            print(
                "_VALID_MUJOCO_GL=",
                getattr(mujoco.gl_context, "_VALID_MUJOCO_GL", None),
            )
            print("_MUJOCO_GL=", getattr(mujoco.gl_context, "_MUJOCO_GL", None))
            print("_SYSTEM=", getattr(mujoco.gl_context, "_SYSTEM", None))
        except Exception as e:
            print("Import/reload gl_context failed for", b, ":", e)
            traceback.print_exc()
            continue

        try:
            m = mujoco.MjModel.from_xml_string(xml)
            d = mujoco.MjData(m)
        except Exception as e:
            print("Failed to create model/data:", e)
            traceback.print_exc()
            continue

        try:
            print("Launching viewer (backend=", b, ")")
            v = mujoco.viewer.launch(m, d)
            print("Viewer launched; sleeping briefly")
            time.sleep(1.0)
            return True
        except Exception as e:
            print("Viewer launch failed for", b, ":", e)
            traceback.print_exc()
            # try next backend
            continue
    return False


def do_offscreen():
    # Create model/data first (fail early if this errors)
    try:
        import mujoco

        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
    except Exception as e:
        print("Failed to create model/data for offscreen:", e)
        traceback.print_exc()
        return False

    print("Attempting offscreen render via mujoco.Renderer")

    # Create renderer (try common constructor signatures)
    renderer = None
    try:
        try:
            renderer = mujoco.Renderer(m, device=0)
        except TypeError:
            renderer = mujoco.Renderer(m)
    except Exception as e:
        print("Could not instantiate mujoco.Renderer:", e)
        traceback.print_exc()
        return False

    rendered = False
    rgb = None
    depth = None

    # Pattern 1: Scene + renderer.render(scene)
    try:
        Scene = getattr(mujoco, "Scene", None)
        if Scene is not None:
            scene = Scene(m, d)
            renderer.render(scene)
            rgb, depth = renderer.read_pixels()
            rendered = True
    except Exception:
        pass

    # Pattern 2: renderer.render(model, data)
    if not rendered:
        try:
            renderer.render(m, d)
            rgb, depth = renderer.read_pixels()
            rendered = True
        except Exception:
            pass

    # Pattern 2b: newer API - update_scene + render()
    if not rendered:
        try:
            if hasattr(renderer, "update_scene") and hasattr(renderer, "render"):
                try:
                    renderer.update_scene(d)
                except TypeError:
                    # some signatures require (data, camera)
                    renderer.update_scene(d, -1)
                out = renderer.render()
                # render() returns rgb or depth depending on flags
                if out is not None:
                    if out.ndim == 3:
                        rgb = out
                        depth = None
                    else:
                        rgb = None
                        depth = out
                    rendered = True
        except Exception:
            pass

    # Pattern 3: renderer.render_scene(scene)
    if not rendered:
        try:
            if hasattr(renderer, "render_scene"):
                Scene = getattr(mujoco, "Scene", None)
                if Scene is not None:
                    sc = Scene(m, d)
                    renderer.render_scene(sc)
                    rgb, depth = renderer.read_pixels()
                    rendered = True
        except Exception:
            pass

    if not rendered:
        try:
            renderer.close()
        except Exception:
            pass
        print("No supported render path on this mujoco build")
        return False

    # Save image
    try:
        from PIL import Image

        img = Image.fromarray(rgb)
        img.save(OUT_PNG)
        print("Wrote", OUT_PNG)
    except Exception as e:
        print("Failed saving image:", e)
        traceback.print_exc()
        try:
            renderer.close()
        except Exception:
            pass
        return False

    try:
        renderer.close()
    except Exception:
        pass
    return True


if __name__ == "__main__":
    print("Initial MUJOCO_GL=", os.environ.get("MUJOCO_GL"))

    # Allow forcing offscreen mode (useful when uv run cannot create main-thread GUI)
    force_offscreen = os.environ.get("FORCE_OFFSCREEN") or os.environ.get("BLOCK_VIEWER")
    if force_offscreen:
        print("FORCE_OFFSCREEN set; skipping interactive viewer and using offscreen")
        if do_offscreen():
            print("Offscreen render succeeded; see", OUT_PNG)
        else:
            print("Offscreen render failed")
    else:
        ok = try_backends()
        if ok:
            print("Interactive viewer worked. Exiting.")
        else:
            print("No interactive backend succeeded; trying offscreen fallback")
            if do_offscreen():
                print("Offscreen render succeeded; see", OUT_PNG)
            else:
                print("All rendering attempts failed")
