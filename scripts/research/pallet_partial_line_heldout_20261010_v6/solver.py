"""Validate unused IMAGE_ROLE lines with both native N3 endpoints held out.

This one structural ablation wraps the frozen v5 point+line solver.  Each
eligible unused line is checked once by the frozen prior-free v4 point solver
using native N3 RGB coordinates.  Its observed line never chooses that pose.
Only a finite available pose with an endpoint RMS above the existing 8 px
threshold removes a line.  Unavailable or uncheckable poses retain the line as
unknown.  No validation projection is turned into an image correspondence.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from ..pallet_cornerwise_independent_20261010_v4 import pose as P
from ..pallet_partial_line_independent_20261010_v5 import solver as V5


POLICY = dict(
    schema="native_N3_both_endpoint_heldout_line_validation_v6",
    validation_coordinates="unchanged native N3 RGB observations only",
    validation_pose_solver="unchanged prior-free unconstrained v4 finite robust point solver",
    validation_robust=True,
    validation_exclusions="actual H union caller temporary exclusions union both edge endpoints",
    active_generator_scoring_refit_excludes_both_endpoints=True,
    observed_line_used_to_choose_validation_pose=False,
    initial_pose_prior=False, initial_projection_prior=False,
    initial_dimension_prior=False, independently_known_dimension=False,
    endpoint_RMS_threshold_px=8.,
    endpoint_RMS="sqrt((d_a**2+d_b**2)/2), signed normal distances to normalized observed line",
    decision="available checkable NEW_POSE RMS<=8: SUPPORTED_RETAIN; RMS>8: CONTRADICTED_REJECT; otherwise UNVERIFIED_RETAIN",
    unavailable_is_no_match=False, unavailable_is_frame_failure=False,
    one_validation_solve_per_original_eligible_unused_edge=True,
    consumed_line_policy="preserve every original raw support line of actually adopted allowed boundary corners; do not validate consumed lines",
    final_input="deep copy of original observation with only contradicted eligible unused edge records removed",
    corner_selection_unchanged=True, validation_projection_becomes_observation=False,
    final_solver="unchanged frozen v5 PointLineBank with the filtered pool",
    final_solver_policy=copy.deepcopy(V5.POLICY),
    temporary_exclusion_is_self_occlusion=False,
    cache="share identical coordinate banks; exact four-ID numeric generators reused across every mask",
    global_cached_generators_may_include_excluded_IDs=True,
    active_validation_candidates_exclude_blocked_IDs=True,
    equivalence="unchanged v4 numerical physical R/t and all-eight model predictions; no excluded observed coordinate read",
    rank_is_local_only=True, global_unique_pose_proven=False,
    line_support_is_not_real_boundary_ownership_certificate=True,
    validation_pose_may_be_wrong_consensus=True,
)


def _delta(after, before):
    return {key: int(after[key] - before[key]) for key in after}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def observation_binding(observation):
    """Semantic bytes only; no filesystem, model, target or image dependency."""
    encoded = json.dumps(observation, sort_keys=True, separators=(",", ":"),
                         allow_nan=True, default=_json_default).encode("utf-8")
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded),
                encoding="sorted compact JSON UTF-8; numpy values converted; nonfinite tokens preserved")


def _raw_edges(observation):
    try:
        return sorted(set(int(line["edge"]) for line in observation.get("lines", [])))
    except (KeyError, TypeError, ValueError):
        # The frozen final solver will declare the invalid observation packet.
        return None


def _matching(bank, points, K, xyz, image_size):
    return (isinstance(bank, P.PoseBank) and bank.known_dimension_index is None
            and np.array_equal(bank.points, np.asarray(points, float), equal_nan=True)
            and np.array_equal(bank.K, np.asarray(K, float))
            and np.array_equal(bank.xyz, np.asarray(xyz, float))
            and bank.image_size == tuple(image_size))


def _witness_problem(solved, blocked):
    """Check stored solver contracts without inspecting blocked image values."""
    if solved.get("available") is not True:
        return "VALIDATION_POSE_UNAVAILABLE:" + str(solved.get("state", "MISSING_STATE"))
    if (solved.get("state") != "NEW_POSE" or solved.get("new_pose_estimated") is not True
            or solved.get("pose_available") is not True or solved.get("no_pose") is not False
            or solved.get("fallback_used") is not False or solved.get("unresolved_ambiguity") is not False):
        return "VALIDATION_POSE_STATUS_INCONSISTENT"
    for key in ("prior_used", "initial_pose_used", "initial_projection_used",
                "initial_dimension_prior_used", "known_dimension_constraint_used",
                "excluded_image_coordinates_used_for_scoring",
                "excluded_image_coordinates_used_for_equivalence",
                "reprojected_points_reused_as_observations"):
        if solved.get(key) is not False:
            return "MISSING_OR_INVALID_PRIOR_FREE_WITNESS:" + key
    try:
        if set(solved["excluded"]) != set(blocked):
            return "VALIDATION_EXCLUSION_WITNESS_MISMATCH"
        used = set(solved["used"])
        if len(used) < 4 or used & set(blocked):
            return "VALIDATION_SCORING_POOL_WITNESS_INVALID"
        for key in ("fit_input_ids", "final_inliers", "generator_ids"):
            ids = set(solved[key])
            if len(ids) < 4 or not ids <= used:
                return "VALIDATION_SELECTED_ID_WITNESS_INVALID:" + key
        candidates = solved["all_candidate_solutions"]
        if not candidates:
            return "VALIDATION_ALL_CANDIDATE_WITNESS_MISSING"
        for candidate in candidates:
            for key in ("generator_ids", "actual_fit_input_ids", "inlier_ids"):
                if not set(candidate[key]) <= used:
                    return "VALIDATION_ACTIVE_CANDIDATE_USES_BLOCKED_ID:" + key
            if len(candidate["residuals_used_px"]) != len(solved["used"]):
                return "VALIDATION_CANDIDATE_SCORING_POOL_WITNESS_INVALID"
        projected = np.asarray(solved["projected"], float)
        if projected.shape != (8, 2) or not np.isfinite(projected).all():
            return "VALIDATION_PROJECTED_WITNESS_INVALID"
        for key, shape in (("R_cf", (3, 3)), ("R_physical", (3, 3)),
                           ("centroid", (3,)), ("cf_extents", (3,))):
            value = np.asarray(solved[key], float)
            if value.shape != shape or not np.isfinite(value).all():
                return "VALIDATION_POSE_WITNESS_INVALID:" + key
    except (KeyError, TypeError, ValueError):
        return "VALIDATION_POSE_WITNESS_MISSING_OR_MALFORMED"
    return None


class EndpointValidatedPointLineBank:
    def __init__(self, points, K, xyz, image_size=(640, 480), *,
                 native_points, native_bank=None, final_bank=None):
        self.final = V5.PointLineBank(points, K, xyz, image_size=image_size, bank=final_bank)
        self.final_bank = self.final.bank
        if native_bank is None:
            if _matching(self.final_bank, native_points, K, xyz, image_size):
                native_bank = self.final_bank
            else:
                native_bank = P.PoseBank(native_points, K, xyz, image_size=image_size)
        if not _matching(native_bank, native_points, K, xyz, image_size):
            raise ValueError("A matching prior-free unconstrained native N3 bank is required")
        if (_matching(self.final_bank, native_points, K, xyz, image_size)
                and native_bank is not self.final_bank):
            raise ValueError("Identical native/final coordinates require the same shared numeric bank")
        self.native_bank = native_bank
        self.native_points = native_bank.points
        self.points, self.K, self.xyz = self.final.points, self.final.K, self.final.xyz

    def solve(self, observation, adopted_boundary_corner_ids=(), hidden=(),
              excluded=(), robust=True):
        H = P._ids(hidden, "hidden")
        temporary = P._ids(excluded, "excluded")
        adopted = P._ids(adopted_boundary_corner_ids, "adopted_boundary_corner_ids")
        blocked_final = set(H) | set(temporary)
        U_final = set(self.final_bank.eligible) - blocked_final
        # Distinct numeric banks are counted once. If native==final, validation
        # and final solve deltas are sequential disjoint intervals of one bank.
        banks = {"native": self.native_bank}
        if self.final_bank is not self.native_bank:
            banks["final"] = self.final_bank
        before = {name: bank.ledger.copy() for name, bank in banks.items()}
        original_binding = observation_binding(observation)
        validation = dict(schema="endpoint_validation_packet_v6", policy=copy.deepcopy(POLICY),
            native_input_hash=self.native_bank.digest,
            final_input_hash=self.final_bank.digest,
            native_points=self.native_points.tolist(), hidden=list(H), temporary_excluded=list(temporary),
            original_observation_binding=original_binding,
            original_line_edges=_raw_edges(observation),
            original_unused_edges=[], retained_edges=[], rejected_edges=[], consumed_edges=[],
            actual_adopted_boundary_ids=[k for k in adopted if k in U_final],
            original_line_contract_checks=[], records=[], attempted_pose_calls=0,
            available_pose_calls=0, unavailable_pose_calls=0, checkable_pose_calls=0,
            unverified_line_decisions=0,
            native_final_bank_shared=self.native_bank is self.final_bank,
            unique_numeric_bank_count=len(banks), bank_before=copy.deepcopy(before),
            bank_aliases={"native": "native", "final": "native" if len(banks) == 1 else "final"},
            observed_line_used_for_validation_pose_choice=False,
            validation_projections_added_to_observations=False,
            actual_self_hidden_separate_from_endpoint_exclusion=True,
            original_corner_selection_and_consumed_support_preserved=True,
            initial_pose_or_dimension_prior_used=False,
            contract_state="VALID", contract_error=None)
        lines, consumed = [], []
        try:
            lines, consumed, checks = self.final._lines(observation, adopted, U_final)
            validation.update(original_unused_edges=[line["edge"] for line in lines],
                              consumed_edges=consumed, original_line_contract_checks=checks)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            # Let the unchanged final solver return its existing contract error.
            # No endpoint solve is attempted from invalid support/identity data.
            validation.update(contract_state="INVALID_OBSERVATION_CONTRACT", contract_error=str(error))
        native_before_validation = self.native_bank.ledger.copy()
        rejected = set()
        for line in lines:
            edge = line["edge"]
            endpoints = V5.EDGES[edge]
            validator_temporary = tuple(sorted(set(temporary) | set(endpoints)))
            blocked = tuple(sorted(set(H) | set(validator_temporary)))
            prior = self.native_bank.ledger.copy()
            # Exactly one solve request even if fewer than four actual N3
            # observations remain. The observed line is not supplied here.
            solved = self.native_bank.solve(excluded=validator_temporary, hidden=H, robust=True)
            after = self.native_bank.ledger.copy()
            problem = _witness_problem(solved, blocked)
            distances, rms = None, None
            if problem is None:
                q = np.asarray(solved["projected"], float)
                distances = q[list(endpoints)] @ line["normal"] - line["offset"]
                rms = float(np.sqrt(np.mean(distances ** 2)))
                if not np.isfinite(rms):
                    problem = "VALIDATION_ENDPOINT_RMS_NONFINITE"
            if problem is not None:
                decision, reason = "UNVERIFIED_RETAIN", problem
            elif rms <= POLICY["endpoint_RMS_threshold_px"]:
                decision, reason = "SUPPORTED_RETAIN", "AVAILABLE_NATIVE_N3_ENDPOINT_RMS_WITHIN_FIXED_8PX"
            else:
                decision, reason = "CONTRADICTED_REJECT", "AVAILABLE_NATIVE_N3_ENDPOINT_RMS_EXCEEDS_FIXED_8PX"
                rejected.add(edge)
            validation["records"].append(dict(edge=edge, endpoints=list(endpoints),
                hidden=list(H), temporary_excluded=list(temporary),
                validator_temporary_excluded=list(validator_temporary), validation_excluded=list(blocked),
                native_input_hash=self.native_bank.digest, attempted_pose_calls=1,
                validation_robust=True, decision=decision, reason=reason,
                endpoint_RMS_px=rms,
                signed_endpoint_distances_px=None if distances is None else distances.tolist(),
                source_line=dict(edge=edge, endpoints=list(endpoints), normal=line["normal"].tolist(),
                                 offset=line["offset"], support_query_ids=list(line["support_query_ids"]),
                                 raw=copy.deepcopy(line["source_line"])),
                solved=solved, bank_before=prior, bank_after=after, bank_delta=_delta(after, prior),
                global_cached_generation_may_include_blocked_IDs=True,
                active_scoring_generators_refits_exclude_blocked_IDs=True))
        validation.update(attempted_pose_calls=len(validation["records"]),
            available_pose_calls=sum(r["solved"].get("available") is True for r in validation["records"]),
            unavailable_pose_calls=sum(r["solved"].get("available") is not True for r in validation["records"]),
            checkable_pose_calls=sum(r["decision"] != "UNVERIFIED_RETAIN" for r in validation["records"]),
            unverified_line_decisions=sum(r["decision"] == "UNVERIFIED_RETAIN" for r in validation["records"]),
            retained_edges=[line["edge"] for line in lines if line["edge"] not in rejected],
            rejected_edges=sorted(rejected),
            validation_bank_before=native_before_validation,
            validation_bank_after=self.native_bank.ledger.copy(),
            validation_bank_delta=_delta(self.native_bank.ledger, native_before_validation))
        filtered = copy.deepcopy(observation)
        if rejected:
            filtered["lines"] = [line for line in filtered.get("lines", []) if int(line["edge"]) not in rejected]
        validation["filtered_observation_binding"] = observation_binding(filtered)
        validation["filtered_raw_line_edges"] = _raw_edges(filtered)
        final_before = self.final_bank.ledger.copy()
        result = self.final.solve(filtered, adopted_boundary_corner_ids=adopted, hidden=H,
                                  excluded=temporary, robust=robust)
        final_after = self.final_bank.ledger.copy()
        validation.update(final_bank_before=final_before, final_bank_after=final_after,
            final_bank_delta=_delta(final_after, final_before),
            bank_after={name: bank.ledger.copy() for name, bank in banks.items()},
            bank_delta={name: _delta(bank.ledger, before[name]) for name, bank in banks.items()},
            original_observation_preserved=observation_binding(observation) == original_binding)
        assert validation["original_observation_preserved"]
        assert not set(consumed) & rejected
        if validation["contract_state"] == "VALID":
            assert result["line_edges"] == validation["retained_edges"]
            assert result["consumed_edges"] == consumed
        result["final_only_operation_counts"] = copy.deepcopy(result["operation_counts"])
        aggregate = {key: sum(delta[key] for delta in validation["bank_delta"].values())
                     for key in self.native_bank.ledger}
        aggregate.update({key: value for key, value in result["operation_counts"].items()
                          if key not in self.native_bank.ledger})
        result.update(operation_counts=aggregate, endpoint_validation=validation,
            endpoint_validation_policy=copy.deepcopy(POLICY),
            endpoint_validation_solver="NATIVE_N3_BOTH_ENDPOINT_HELDOUT_THEN_FROZEN_V5_C2",
            endpoint_validation_observed_line_used_for_pose_choice=False,
            endpoint_validation_projection_used_as_observation=False)
        return result
