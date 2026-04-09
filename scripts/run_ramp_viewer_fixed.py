#!/usr/bin/env python3
"""ランプ登坂デバッグビューワ起動スクリプト（固定版）

usage:
  python3 scripts/run_ramp_viewer_fixed.py --mode debug --target seeker --episodes 1
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import math
import time

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="debug")
    p.add_argument("--target", default="seeker", choices=["seeker", "hider"])
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model-path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    import scripts.verify.run_ramp_viewer as helper
    # delegate to helper which builds env and runs viewer
    helper_main = helper.main
    helper_main()


if __name__ == "__main__":
    main()
