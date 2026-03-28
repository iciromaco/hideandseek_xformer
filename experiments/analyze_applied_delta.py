import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

JPATH = "experiments/applied_forward_dynamics.json"
OUT_DIR = "experiments/plots"


def mad(x):
    return np.median(np.abs(x - np.median(x)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    j = json.load(open(JPATH))
    recs = j.get("records_sample", [])
    if not recs:
        print("no records in", JPATH)
        return
    arr_ap = np.array([r["applied_forward"] for r in recs], dtype=np.float32)
    arr_dv = np.array([r["delta_vx"] for r in recs], dtype=np.float32)

    # raw correlation
    corr_raw = float(np.corrcoef(arr_ap, arr_dv)[0, 1]) if arr_ap.size > 1 else 0.0

    # robust outlier removal via MAD
    med = np.median(arr_dv)
    madv = mad(arr_dv)
    thresh = 6.0 * (madv if madv > 1e-6 else np.std(arr_dv))
    mask = np.abs(arr_dv - med) <= thresh
    arr_ap_f = arr_ap[mask]
    arr_dv_f = arr_dv[mask]

    corr_filt = float(np.corrcoef(arr_ap_f, arr_dv_f)[0, 1]) if arr_ap_f.size > 1 else 0.0

    # linear fit on filtered
    if arr_ap_f.size > 1:
        coef = np.polyfit(arr_ap_f, arr_dv_f, 1)
    else:
        coef = [0.0, 0.0]

    # plots
    plt.figure(figsize=(6, 6))
    plt.scatter(arr_ap, arr_dv, s=8, alpha=0.3, label="raw")
    plt.scatter(arr_ap_f, arr_dv_f, s=10, alpha=0.8, label="filtered")
    xs = np.linspace(arr_ap.min(), arr_ap.max(), 200)
    ys = np.polyval(coef, xs)
    plt.plot(xs, ys, "r-", lw=2, label=f"lin fit: {coef[0]:.3f}x+{coef[1]:.3f}")
    plt.xlabel("applied_forward")
    plt.ylabel("delta_vx")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "applied_vs_delta_filtered.png")
    plt.savefig(out_png)
    plt.close()

    summary = {
        "corr_raw": corr_raw,
        "corr_filtered": corr_filt,
        "n_raw": int(arr_ap.size),
        "n_filtered": int(arr_ap_f.size),
        "fit_coef": coef.tolist(),
        "median_delta": float(med),
        "mad_delta": float(madv),
    }
    with open(os.path.join(OUT_DIR, "applied_delta_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved", out_png)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
