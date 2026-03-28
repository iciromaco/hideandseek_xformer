import faulthandler
import os
import traceback

faulthandler.enable()

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "cgl")

try:
    import mujoco

    print(
        "mujoco",
        getattr(mujoco, "__version__", "unknown"),
        "MUJOCO_GL=",
        os.environ.get("MUJOCO_GL"),
    )
    # print top-level attributes
    print("mujoco module attrs:", [a for a in dir(mujoco) if not a.startswith("__")])

    m = mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.01"/><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>')
    d = mujoco.MjData(m)
    print("Model/data created")

    try:
        # Try renderer introspection
        try:
            r = None
            try:
                r = mujoco.Renderer(m, device=0)
            except TypeError:
                r = mujoco.Renderer(m)
            print("Renderer instantiated:", type(r))
            print("Renderer dir:", [a for a in dir(r) if not a.startswith("__")])
            print("Has _gl_context?", hasattr(r, "_gl_context"))
            try:
                r.close()
                print("Renderer closed OK")
            except Exception as e:
                print("Renderer.close failed:", e)
        except Exception as e:
            print("Renderer instantiation failed:", e)
            traceback.print_exc()

    except Exception:
        traceback.print_exc()

    # Try launching viewer (may raise native exceptions)
    try:
        import mujoco.viewer as V

        print("viewer attrs:", [a for a in dir(V) if not a.startswith("__")])
        print("Attempting mujoco.viewer.launch...")
        mujoco.viewer.launch(m, d)
    except Exception:
        print("viewer.launch raised:")
        traceback.print_exc()
        raise

except Exception:
    traceback.print_exc()
    raise
