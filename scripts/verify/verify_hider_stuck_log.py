import argparse
import re
from pathlib import Path


def parse_debug_chunks(text: str):
    pattern = re.compile(r"\[HNS_DEBUG\].*?(?=\[HNS_DEBUG\]|Step:|$)", re.DOTALL)
    return pattern.findall(text)


def extract_fields(chunk: str):
    def get_int(name: str, default: int = -1) -> int:
        m = re.search(rf"{name}=(-?\d+)", chunk)
        return int(m.group(1)) if m else default

    def get_float(name: str, default: float = -1.0) -> float:
        m = re.search(rf"{name}=(-?\d+(?:\.\d+)?)", chunk)
        return float(m.group(1)) if m else default

    h2_ctrl_match = re.search(r"h2_ctrl=\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)", chunk)
    h2_ctrl_x = float(h2_ctrl_match.group(1)) if h2_ctrl_match else 0.0
    h2_ctrl_y = float(h2_ctrl_match.group(2)) if h2_ctrl_match else 0.0

    return {
        "step": get_int("step"),
        "prep": get_int("prep"),
        "dpos_h1": get_float("dpos_h1"),
        "dpos_h2": get_float("dpos_h2"),
        "still_h1": get_int("still_h1"),
        "still_h2": get_int("still_h2"),
        "stuck_h1": get_int("stuck_h1"),
        "stuck_h2": get_int("stuck_h2"),
        "h2_ctrl_x": h2_ctrl_x,
        "h2_ctrl_y": h2_ctrl_y,
    }


def build_stuck_segments(rows, stuck_key: str):
    segments = []
    active = None
    prev_row = None

    for row in rows:
        is_stuck = row.get(stuck_key, 0) == 1

        if is_stuck:
            if active is None:
                active = {
                    "start_step": row["step"],
                    "end_step": row["step"],
                    "count": 1,
                    "max_still": row["still_h1"] if stuck_key == "stuck_h1" else row["still_h2"],
                }
            else:
                same_episode_progress = prev_row is not None and row["step"] >= prev_row["step"]
                if same_episode_progress:
                    active["end_step"] = row["step"]
                    active["count"] += 1
                    still_now = row["still_h1"] if stuck_key == "stuck_h1" else row["still_h2"]
                    if still_now > active["max_still"]:
                        active["max_still"] = still_now
                else:
                    segments.append(active)
                    active = {
                        "start_step": row["step"],
                        "end_step": row["step"],
                        "count": 1,
                        "max_still": row["still_h1"] if stuck_key == "stuck_h1" else row["still_h2"],
                    }
        else:
            if active is not None:
                segments.append(active)
                active = None

        prev_row = row

    if active is not None:
        segments.append(active)

    return segments


def find_segment_starts(rows, stuck_key: str):
    starts = []
    prev_stuck = 0
    prev_row = None

    for row in rows:
        cur_stuck = 1 if row.get(stuck_key, 0) == 1 else 0
        episode_reset = prev_row is not None and row["step"] < prev_row["step"]
        if cur_stuck == 1 and (prev_stuck == 0 or episode_reset):
            starts.append(row)
        prev_stuck = cur_stuck
        prev_row = row

    return starts


def print_stuck_events(rows, max_events: int):
    def _emit(agent_key: str, stuck_key: str, still_key: str, dpos_key: str):
        printed = 0
        prev_row = None
        prev_stuck = 0

        for row in rows:
            cur_stuck = 1 if row.get(stuck_key, 0) == 1 else 0
            episode_reset = prev_row is not None and row["step"] < prev_row["step"]
            is_start = cur_stuck == 1 and (prev_stuck == 0 or episode_reset)

            if is_start:
                prep_prev = prev_row["prep"] if prev_row is not None else -1
                dpos_prev = prev_row[dpos_key] if prev_row is not None else -1.0
                still_prev = prev_row[still_key] if prev_row is not None else -1
                prep_changed = int(prev_row is not None and row["prep"] != prev_row["prep"])

                print(
                    f"event agent={agent_key} step={row['step']} prep_prev={prep_prev} prep={row['prep']} prep_changed={prep_changed} "
                    f"dpos_prev={dpos_prev:.4f} dpos_now={row[dpos_key]:.4f} "
                    f"still_prev={still_prev} still_now={row[still_key]} "
                    f"h2_ctrl=({row['h2_ctrl_x']:.3f},{row['h2_ctrl_y']:.3f})"
                )
                printed += 1
                if printed >= max_events:
                    break

            prev_stuck = cur_stuck
            prev_row = row

    print("stuck_events:")
    _emit("h1", "stuck_h1", "still_h1", "dpos_h1")
    _emit("h2", "stuck_h2", "still_h2", "dpos_h2")


def print_context_before(rows, stuck_key: str, still_key: str, before_steps: int, max_segments: int):
    starts = find_segment_starts(rows, stuck_key)
    if not starts:
        return

    print(f"context_{stuck_key}: step prep dpos_h1 dpos_h2 still_h1 still_h2 stuck_h1 stuck_h2")
    shown = 0
    for start in starts:
        if shown >= max_segments:
            break

        window = []
        for row in rows:
            if row["step"] >= 0 and row["step"] <= start["step"] and row["step"] >= start["step"] - before_steps:
                window.append(row)

        if not window:
            continue

        print(f"-- segment_start_step={start['step']} key={stuck_key} --")
        for row in window:
            print(
                f"{row['step']:4d} {row['prep']:4d} "
                f"{row['dpos_h1']:.4f} {row['dpos_h2']:.4f} "
                f"{row['still_h1']:8d} {row['still_h2']:8d} "
                f"{row['stuck_h1']:8d} {row['stuck_h2']:8d}"
            )
        shown += 1


def main():
    parser = argparse.ArgumentParser(description="Extract stuck/still debug info from HNS logs")
    parser.add_argument("log_path", type=Path, help="Path to captured terminal log file")
    parser.add_argument("--max", type=int, default=40, dest="max_rows", help="Max rows to print")
    parser.add_argument("--all", action="store_true", help="Print all parsed debug rows (not only stuck ones)")
    parser.add_argument("--segments", action="store_true", help="Print contiguous stuck segments summary")
    parser.add_argument("--segments-max", type=int, default=20, help="Max segment rows per agent")
    parser.add_argument("--context-before", type=int, default=0, help="Print N steps before each stuck segment start")
    parser.add_argument("--context-max-segments", type=int, default=10, help="Max segment contexts per agent")
    parser.add_argument("--events", action="store_true", help="Print one-line transition summary at stuck starts")
    parser.add_argument("--events-max", type=int, default=20, help="Max stuck-start events per agent")
    args = parser.parse_args()

    if not args.log_path.exists():
        raise FileNotFoundError(f"Log file not found: {args.log_path}")

    text = args.log_path.read_text(encoding="utf-8", errors="ignore")
    chunks = parse_debug_chunks(text)
    rows = [extract_fields(c) for c in chunks]

    valid_rows = [r for r in rows if r["step"] >= 0]
    stuck_rows = [r for r in valid_rows if r["stuck_h1"] == 1 or r["stuck_h2"] == 1]

    print(f"parsed_debug_rows={len(valid_rows)}")
    print(f"stuck_rows={len(stuck_rows)}")

    if args.segments:
        h1_segments = build_stuck_segments(valid_rows, "stuck_h1")
        h2_segments = build_stuck_segments(valid_rows, "stuck_h2")
        print(f"stuck_segments_h1={len(h1_segments)}")
        print(f"stuck_segments_h2={len(h2_segments)}")

        if h1_segments:
            print("segments_h1: start_step end_step count max_still")
            for seg in h1_segments[: args.segments_max]:
                print(
                    f"{seg['start_step']:9d} {seg['end_step']:8d} "
                    f"{seg['count']:5d} {seg['max_still']:9d}"
                )

        if h2_segments:
            print("segments_h2: start_step end_step count max_still")
            for seg in h2_segments[: args.segments_max]:
                print(
                    f"{seg['start_step']:9d} {seg['end_step']:8d} "
                    f"{seg['count']:5d} {seg['max_still']:9d}"
                )

    if args.context_before > 0:
        print_context_before(
            valid_rows,
            stuck_key="stuck_h1",
            still_key="still_h1",
            before_steps=args.context_before,
            max_segments=args.context_max_segments,
        )

    if args.events:
        print_stuck_events(valid_rows, max_events=args.events_max)
        print_context_before(
            valid_rows,
            stuck_key="stuck_h2",
            still_key="still_h2",
            before_steps=args.context_before,
            max_segments=args.context_max_segments,
        )

    display_rows = valid_rows if args.all else stuck_rows
    if not display_rows:
        print("no_rows_to_display")
        return

    print("step prep dpos_h1 dpos_h2 still_h1 still_h2 stuck_h1 stuck_h2")
    for row in display_rows[: args.max_rows]:
        print(
            f"{row['step']:4d} {row['prep']:4d} "
            f"{row['dpos_h1']:.4f} {row['dpos_h2']:.4f} "
            f"{row['still_h1']:8d} {row['still_h2']:8d} "
            f"{row['stuck_h1']:8d} {row['stuck_h2']:8d}"
        )


if __name__ == "__main__":
    main()
