# calib/motion_models.py
"""
Forward-motion time model (piecewise accelerate → cruise).

This module exposes a single public API:
    - fwd_sec_from_offset_piecewise(offset_m: float, **overrides) -> float

It converts an alignment offset (meters) into a forward command duration (seconds)
by using a kinematic piecewise model fitted from logs that contain only:
`t_monotonic` (time) and `dist_z` (depth-to-pallet).

Model:
    Let d be the forward travel distance (m) we want to achieve.

    Params: t0 (latency), t1 (accel duration), a (acceleration)
        vmax = a * t1
        d_acc = 0.5 * a * t1^2

    Time to travel distance d:
        if d <= d_acc:
            t(d) = t0 + sqrt(2*d/a)
        else:
            t(d) = t0 + t1 + (d - d_acc) / vmax

    Finally, clamp to [min_sec, max_sec] and allow field tuning via:
        d_eff = scale * |offset| + bias

Defaults are read from calib.config, but every parameter can be overridden
via function keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    # Local package config (expected path in this repository)
    from . import config as cfg
except Exception:  # pragma: no cover
    # Fallback defaults if imported standalone (useful for unit tests)
    class _Cfg:
        FWD_T0 = 0.0
        FWD_T1 = 3.0
        FWD_A = 0.1
        FWD_SCALE = 1.0
        FWD_BIAS = 0.0
        FWD_MIN_SEC = 1.0
        FWD_MAX_SEC = 15.0
    cfg = _Cfg()  # type: ignore


@dataclass(frozen=True)
class PiecewiseFwdParams:
    """Parameters for the accelerate→cruise forward-time model."""
    t0: float = cfg.FWD_T0          # latency (s)
    t1: float = cfg.FWD_T1          # accel duration (s)
    a: float = cfg.FWD_A            # acceleration (m/s^2)
    scale: float = cfg.FWD_SCALE    # d_eff = scale*|offset| + bias
    bias: float = cfg.FWD_BIAS      # meters
    min_sec: float = cfg.FWD_MIN_SEC
    max_sec: float = cfg.FWD_MAX_SEC

    @property
    def vmax(self) -> float:
        return self.a * self.t1

    @property
    def d_acc(self) -> float:
        return 0.5 * self.a * (self.t1 ** 2)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_sqrt(x: float) -> float:
    return (x if x > 0 else 0) ** 0.5


def time_from_distance_piecewise(d: float, params: PiecewiseFwdParams) -> float:
    """
    Convert a physical travel distance (m) to time (s) using piecewise kinematics.

    Args:
        d: desired forward travel distance in meters (>= 0).
        params: piecewise model parameters.

    Returns:
        Time in seconds, clamped to [min_sec, max_sec].
    """
    d = max(0.0, d)
    # Derived quantities
    d_acc = params.d_acc
    vmax = max(1e-6, params.vmax)  # avoid division by tiny/zero

    if d <= d_acc:
        t_cmd = params.t0 + _safe_sqrt(2.0 * d / max(1e-9, params.a))
    else:
        t_cmd = params.t0 + params.t1 + (d - d_acc) / vmax

    return _clamp(t_cmd, params.min_sec, params.max_sec)


def fwd_sec_from_offset_piecewise(
    offset_m: float,
    *,
    t0: Optional[float] = None,
    t1: Optional[float] = None,
    a: Optional[float] = None,
    scale: Optional[float] = None,
    bias: Optional[float] = None,
    min_sec: Optional[float] = None,
    max_sec: Optional[float] = None,
) -> float:
    """
    Convert |offset_smooth| (m) from OFFSET_CHECK into a forward duration (s).

    By default, loads parameters from calib.config. Each parameter can be
    overridden via keyword arguments for experimentation or A/B testing.

    Args:
        offset_m: absolute lateral offset in meters measured at OFFSET_CHECK.
        t0, t1, a: kinematic parameters (latency, accel duration, acceleration).
        scale, bias: field-tuning to map offset to effective travel distance.
        min_sec, max_sec: safety clamps for command duration.

    Returns:
        Forward duration in seconds (float).
    """
    # Load defaults from config, then apply overrides
    params = PiecewiseFwdParams(
        t0=cfg.FWD_T0 if t0 is None else t0,
        t1=cfg.FWD_T1 if t1 is None else t1,
        a=cfg.FWD_A if a is None else a,
        scale=cfg.FWD_SCALE if scale is None else scale,
        bias=cfg.FWD_BIAS if bias is None else bias,
        min_sec=cfg.FWD_MIN_SEC if min_sec is None else min_sec,
        max_sec=cfg.FWD_MAX_SEC if max_sec is None else max_sec,
    )

    # Map lateral offset to effective forward distance
    d_eff = max(0.0, params.scale * abs(offset_m) + params.bias)

    return time_from_distance_piecewise(d_eff, params)
