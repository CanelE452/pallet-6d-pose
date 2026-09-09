"""Pure frame-slider conversions for the manual annotation editor."""

from __future__ import annotations


# Qt/OpenCV cannot reliably change a trackbar maximum after creation, so the
# frame slider uses one fixed-resolution position range for every session.
_FRAME_SLIDER_TICKS = 500


def _frame_cur_to_tick(current, total):
    return (0 if total <= 1 else
            int(round(current / (total - 1) * _FRAME_SLIDER_TICKS)))


def _frame_tick_to_cur(tick, total):
    return (0 if total <= 1 else
            int(round(tick / _FRAME_SLIDER_TICKS * (total - 1))))


def _frame_trackbar_target(current, total, tick):
    """Return a user-requested frame, never a lossy programmatic round-trip.

    Large incoming sessions contain far more frames than the fixed 500 slider
    ticks, so adjacent frames often map to the same tick. Converting that
    unchanged tick back to a frame would roll sequential navigation back to a
    representative frame. Only a raw tick change can be a slider request.
    """
    if total <= 1 or int(tick) == _frame_cur_to_tick(current, total):
        return None
    target = _frame_tick_to_cur(int(tick), total)
    return None if target == current else target
