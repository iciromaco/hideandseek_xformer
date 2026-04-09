"""
Package marker for experiments.

This file makes the `experiments` directory an importable package,
preventing mypy from seeing the same source file under two module
names (e.g. "run_step_response_nowalls" and "experiments.run_step_response_nowalls").
"""

# intentionally empty
