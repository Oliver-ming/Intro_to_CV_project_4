"""Iteration events for the one 30k closed-loop run and the matched control."""

from __future__ import annotations

REPORT_ITERS = (7000, 30000)
CONTROL_ITERS = (1500, 3000, 4500, 7000, 30000)


def closed_loop_event(iteration: int) -> str | None:
    """Return the action at this iteration, or None when training just continues."""
    if iteration == 1500:
        return "insert_round_1"
    if iteration == 3000:
        return "record_round_1_and_insert_round_2"
    if iteration == 4500:
        return "record_round_2"
    if iteration in REPORT_ITERS:
        return "report"
    return None
