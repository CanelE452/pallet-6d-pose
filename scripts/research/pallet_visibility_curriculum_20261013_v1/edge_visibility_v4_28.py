"""Pure opt-in visibility/support diagnostics for proposed B/I image edges.

An empty target mask is UNKNOWN, never an occlusion label. Only a separately
bound non-target occluder mask can support an occluded gap. This module does
not query RGB, infer masks, change states/points or admit line/KP/pose factors.
It provides a pre-sampling predicate so known occluder boundaries cannot be
used as pallet-edge pixels. The original target mask is never cut into a new
artificial edge. All local probes are gated before mask lookup.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import image_supported_edge_chains_v4_17c as NMS
from .interior_observation import NEIGHBORS


SCHEMA = 'CONDITIONAL_EDGE_VISIBILITY_SUPPORT_DIAGNOSTICS_V4_28'
OBSERVED_SUPPORT = 'observed_support'
OCCLUDER_SUPPORTED_GAP = 'occluder_supported_gap'
UNSUPPORTED_UNKNOWN = 'unsupported_unknown'
CONTEXT_PX = float(NMS.DEFAULT_CONFIGURATION['boundary_context_px'])
UNKNOWN_ENDPOINT_EXCLUSION_PX = 2. * CONTEXT_PX
MASK_THRESHOLD = .5


def _id(k):
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 0 <= k < 8:
        raise ValueError('Explicit native endpoint index required')
    return int(k)


def _finite_xy(value, name):
    a = np.asarray(value, float)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError('Finite Nx2 ' + name + ' required')
    return a


def _occupancy(value, image_hw, name):
    a = np.asarray(value, float)
    if a.shape != image_hw or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError('Bound finite0..1 ' + name + ' required')
    a = a.copy()
    a.setflags(write=False)
    return a


def mask_pixel_sha256(array):
    """Canonical float64 occupancy-array binding, matching the topology cache."""
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()


def _clip_segment(a, b, image_hw):
    """Liang-Barsky clipping of this segment only, with no invented closure."""
    h, w = image_hw
    d = b-a
    lo, hi = 0., 1.
    for p, q in ((-d[0], a[0]), (d[0], w-1-a[0]), (-d[1], a[1]), (d[1], h-1-a[1])):
        if abs(p) < 1e-15:
            if q < 0: return None
            continue
        r = q/p
        if p < 0: lo = max(lo, r)
        else: hi = min(hi, r)
        if lo > hi: return None
    # Floating cancellation for extremely distant model endpoints can exceed
    # the analytic canvas bound by a few ulps. Clamp sampling endpoints only;
    # the original projected model segment remains unchanged in provenance.
    high = np.array([w-1,h-1],float)
    return np.clip(a+lo*d,0.,high), np.clip(a+hi*d,0.,high)


class EdgeVisibilityContext:
    """Immutable mask snapshots plus permanent state-gated model directions.

    ``target_mask`` remains unchanged. A fitting caller MUST apply
    ``query_allowed`` to candidate lattice sites before any RGB/mask remap.
    ``context_radius_px`` bounds every downstream probe around a site; the
    independent occluder is expanded by that footprint, and the original
    unknown-endpoint14px disk is protected by14+footprint at its center.
    """
    def __init__(self, target_mask, geometry, states, visible_measurements, *,
                 image_binding, occluder_mask=None, occluder_binding=None):
        self.states = tuple(NMS.V3._state(s) for s in states)
        self.hidden = tuple(bool(v) for v in geometry['initial_self_hidden'])
        self.image_hw = tuple(geometry['image_hw'])
        if (len(self.states) != 8 or len(self.hidden) != 8
                or self.hidden != tuple(s == 'S' for s in self.states)
                or len(self.image_hw) != 2 or any(isinstance(v, bool) or int(v) != v or v <= 0 for v in self.image_hw)):
            raise ValueError('Exact permanent initialS/eight states and positive canvas required')
        if not isinstance(image_binding, Mapping) or tuple(image_binding.get('image_hw', ())) != self.image_hw:
            raise ValueError('Exact source-image/canvas binding required')
        NMS.D._sha(image_binding.get('sha256'), 'source image')
        self.image_binding = deepcopy(dict(image_binding))
        self.target_mask = _occupancy(target_mask, self.image_hw, 'target occupancy mask')
        self.occluder_binding = None
        if occluder_mask is None:
            if occluder_binding is not None:
                raise ValueError('Occluder binding without a mask is unsupported')
            self.occluder_mask = None
        else:
            if (not isinstance(occluder_binding, Mapping)
                    or occluder_binding.get('image_sha256') != image_binding['sha256']
                    or tuple(occluder_binding.get('image_hw', ())) != self.image_hw
                    or occluder_binding.get('independent_of_target_mask') is not True
                    or occluder_binding.get('mask_role') != 'NON_TARGET_OCCLUDER'):
                raise ValueError('Independently sourced non-target occluder mask binding required')
            NMS.D._sha(occluder_binding.get('sha256'), 'independent occluder mask')
            self.occluder_mask = _occupancy(occluder_mask, self.image_hw, 'independent occluder mask')
            if occluder_binding.get('mask_pixel_sha256') != mask_pixel_sha256(self.occluder_mask):
                raise ValueError('Independent occluder actual occupancy-array digest differs')
            self.occluder_binding = deepcopy(dict(occluder_binding))
        # All model points are geometry-only. Never touch an S model slot or
        # any U/O/T observation record; only nonS model endpoints are cached.
        self.projection = {}
        for k in range(8):
            if self.hidden[k] or self.states[k] == 'S': continue
            try:
                q = np.asarray(geometry['initial_projection_xy'][k], float)
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if q.shape == (2,) and np.isfinite(q).all(): self.projection[k] = q.copy()
        self.anchors = set()
        for record in visible_measurements:
            k = _id(record['native_corner'])
            if self.hidden[k] or self.states[k] not in ('B', 'I'): continue
            q = np.asarray(record['xy'], float)
            if (q.shape == (2,) and np.isfinite(q).all() and 0 <= q[0] < self.image_hw[1]
                    and 0 <= q[1] < self.image_hw[0]):
                if k in self.anchors: raise ValueError('Unique visible B/I anchor records required')
                self.anchors.add(k)
        self._expanded_occluder = {}

    def _eligible(self, k, mate):
        k, mate = _id(k), _id(mate)
        # This precedes model endpoints, candidate xy, observed chains and masks.
        if self.hidden[k] or self.hidden[mate] or self.states[k] not in ('B', 'I') or self.states[mate] == 'S':
            return False, 'initialS_or_nonBI_anchor'
        if mate not in NEIGHBORS[k]: return False, 'not_incident_native_edge'
        if k not in self.anchors: return False, 'missing_finite_inframe_BI_anchor'
        if k not in self.projection or mate not in self.projection: return False, 'missing_finite_nonS_model_endpoint'
        if np.linalg.norm(self.projection[mate]-self.projection[k]) <= 1e-10:
            return False, 'degenerate_model_segment'
        return True, None

    def _base_gate(self, k, mate, xy, radius):
        h, w = self.image_hw
        keep = ((xy[:, 0] >= radius) & (xy[:, 1] >= radius)
                & (xy[:, 0] <= w-1-radius) & (xy[:, 1] <= h-1-radius))
        if self.states[mate] in ('U', 'O', 'T'):
            keep &= np.linalg.norm(xy-self.projection[mate], axis=1) >= UNKNOWN_ENDPOINT_EXCLUSION_PX+radius
        return keep

    def _occluder_blocked(self, xy, radius):
        if self.occluder_mask is None or not len(xy): return np.zeros(len(xy), bool)
        # Square footprint safely covers every RGB/NMS/mask probe within r.
        r = int(np.ceil(radius))
        if r not in self._expanded_occluder:
            positive = (self.occluder_mask >= MASK_THRESHOLD).astype(np.uint8)
            blocked = cv2.dilate(positive, np.ones((2*r+1, 2*r+1), np.uint8)) if r else positive
            self._expanded_occluder[r] = blocked
        nearest = np.rint(xy).astype(int)
        return self._expanded_occluder[r][nearest[:, 1], nearest[:, 0]].astype(bool)

    def query_allowed(self, native_corner, native_neighbor, xy, *, context_radius_px=CONTEXT_PX):
        """Pre-remap boolean gate; no target-mask values are queried here.

        On an excluded S/U/O/T anchor or S mate this returns scalarFalse before
        even converting ``xy``. Eligible callers receive a lengthN bool array.
        """
        k, mate = _id(native_corner), _id(native_neighbor)
        eligible, _ = self._eligible(k, mate)
        if not eligible: return False
        radius = float(context_radius_px)
        if not np.isfinite(radius) or radius < 0: raise ValueError('Finite nonnegative probe footprint required')
        points = _finite_xy(xy, 'candidate sampling sites')
        keep = self._base_gate(k, mate, points, radius)
        indices = np.flatnonzero(keep)
        if len(indices): keep[indices] &= ~self._occluder_blocked(points[indices], radius)
        return keep

    def gated_sampling_inputs(self):
        """Unmodified read-only target occupancy plus mandatory site predicate."""
        return dict(target_mask=self.target_mask, query_allowed=self.query_allowed,
            independent_occluder_present=self.occluder_mask is not None,
            target_mask_not_cut_or_complemented=True,
            apply_query_allowed_before_all_RGB_and_mask_remaps=True)

    def classify_expected_edge(self, native_corner, native_neighbor, *, observed_chain_points=None,
                               mask_context_kind=None):
        """Describe model-segment coverage; never classify a keypoint as O/B/U.

        ``observed_chain_points`` must be actual supported inliers from the
        caller's image observer, not interpolated model points. Support here
        means proximity plus target context, not certified native identity.
        """
        k, mate = _id(native_corner), _id(native_neighbor)
        eligible, reason = self._eligible(k, mate)
        base = dict(schema=SCHEMA, native_corner=k, native_neighbor=mate, eligible=eligible,
            classes=[OBSERVED_SUPPORT, OCCLUDER_SUPPORTED_GAP, UNSUPPORTED_UNKNOWN],
            target_mask_absence_is_not_occlusion=True, independent_occluder_present=self.occluder_mask is not None,
            occluder_supported_gap_is_conditional_on_model_segment=True,
            state_changed=False, full_KP_or_U_intersection_created=False, actual_native_identity_certified=False,
            S_U_observation_coordinate_or_local_queries=0, RGB_queries=0, new_NN_or_pose_calls=0)
        if not eligible: return dict(base, reason=reason, samples=[], counts={name:0 for name in base['classes']})
        if mask_context_kind not in (None, 'TARGET_BACKGROUND_OUTLINE', 'TARGET_TARGET_INTERNAL'):
            raise ValueError('Explicit supported target context required')
        a, b = self.projection[k], self.projection[mate]
        clipped = _clip_segment(a, b, self.image_hw)
        if clipped is None: return dict(base, reason='model_segment_outside_canvas', samples=[], counts={name:0 for name in base['classes']})
        aa, bb = clipped
        length = float(np.linalg.norm(bb-aa))
        samples = aa+np.linspace(0.,1.,max(2,int(np.ceil(length))+1))[:,None]*(bb-aa)
        direction = (b-a)/np.linalg.norm(b-a)
        normal = np.array([-direction[1], direction[0]])
        # Both actual context probes must pass permanent endpoint/canvas gates
        # before even an independent-occluder or target-mask lookup.
        plus, minus = samples+CONTEXT_PX*normal, samples-CONTEXT_PX*normal
        allowed = (self._base_gate(k,mate,samples,0.) & self._base_gate(k,mate,plus,0.)
                   & self._base_gate(k,mate,minus,0.))
        classes = np.full(len(samples), UNSUPPORTED_UNKNOWN, dtype=object)
        occ = np.zeros(len(samples), bool)
        ii = np.flatnonzero(allowed)
        if len(ii): occ[ii] = self._occluder_blocked(samples[ii], 0.)
        classes[allowed & occ] = OCCLUDER_SUPPORTED_GAP
        fit_allowed = self.query_allowed(k,mate,samples,context_radius_px=CONTEXT_PX)
        actual = np.empty((0,2)) if observed_chain_points is None else _finite_xy(observed_chain_points,'actual supported chain points')
        input_count = len(actual)
        if len(actual):
            h,w = self.image_hw
            if np.any(actual < 0) or np.any(actual[:,0] >= w) or np.any(actual[:,1] >= h):
                raise ValueError('Actual chain inliers must be on the bound image canvas')
            actual = actual[self.query_allowed(k,mate,actual,context_radius_px=CONTEXT_PX)]
        if len(actual):
            distances, _ = cKDTree(actual).query(samples)
        else: distances = np.full(len(samples), np.inf)
        context_pass = np.zeros(len(samples), bool)
        jj = np.flatnonzero(fit_allowed)
        if len(jj):
            pidx, midx = np.rint(plus[jj]).astype(int), np.rint(minus[jj]).astype(int)
            p, m = self.target_mask[pidx[:,1],pidx[:,0]], self.target_mask[midx[:,1],midx[:,0]]
            cfg = NMS.DEFAULT_CONFIGURATION
            if mask_context_kind == 'TARGET_TARGET_INTERNAL':
                good = (p >= cfg['minimum_inside']) & (m >= cfg['minimum_inside'])
            elif mask_context_kind == 'TARGET_BACKGROUND_OUTLINE':
                good = ((np.maximum(p,m) >= cfg['minimum_inside']) & (np.minimum(p,m) <= cfg['maximum_outside'])
                        & (np.abs(p-m) >= cfg['minimum_transition']))
            else: good = np.maximum(p,m) >= MASK_THRESHOLD
            context_pass[jj] = good
        supported = fit_allowed & context_pass & (distances <= CONTEXT_PX)
        classes[supported] = OBSERVED_SUPPORT
        counts = {name:int(np.count_nonzero(classes == name)) for name in base['classes']}
        return dict(base, reason=None, model_segment_xy=[a.tolist(),b.tolist()],
            model_segment_used_for_direction_not_image_measurement=True,
            model_segment_clipped_for_sampling_only=True, mask_context_kind=mask_context_kind,
            samples=[dict(xy=q.tolist(),classification=str(label),sampling_eligible=bool(ok),
                          fitting_query_allowed=bool(fit),actual_chain_distance_px=float(d) if np.isfinite(d) else None)
                     for q,label,ok,fit,d in zip(samples,classes,allowed,fit_allowed,distances)],
            counts=counts, actual_chain_inlier_input_count=input_count, actual_chain_inlier_count=len(actual),
            target_mask_pixel_digest=mask_pixel_sha256(self.target_mask),
            independent_occluder_binding=deepcopy(self.occluder_binding),
            no_unsupported_gap_promoted_to_actual_edge=True,
            support_distance_is_existing_context_px=CONTEXT_PX,
            unknown_endpoint_exclusion_px=UNKNOWN_ENDPOINT_EXCLUSION_PX,
            fitting_guard_includes_full_probe_footprint=True)


def prepare_edge_visibility(target_mask, geometry, states, visible_measurements, *, image_binding,
                            occluder_mask=None, occluder_binding=None):
    return EdgeVisibilityContext(target_mask,geometry,states,visible_measurements,
        image_binding=image_binding,occluder_mask=occluder_mask,occluder_binding=occluder_binding)
