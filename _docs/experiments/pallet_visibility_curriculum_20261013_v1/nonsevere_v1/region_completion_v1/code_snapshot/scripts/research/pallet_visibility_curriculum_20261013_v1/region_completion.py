"""Finite P3P hypotheses with protected visible points and coarse image regions.

Three precise image points can generate several full-pose roots. Coarse hidden
predictions rank those roots without becoming precise point correspondences.
This opt-in policy is distinct from V1's translation-only three-point branch.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import combinations

import cv2
import numpy as np

from . import anchored_completion as A
from . import boundary_rules as B
from . import curriculum as M


@dataclass(frozen=True)
class RegionCompletionConfig:
    region_weight: float = .10
    include_p3p: bool = True
    maximum_roots: int = 4


def p3p_candidates(xyz, points, K, anchor_ids):
    """Choose one spread-maximizing native triple, then enumerate its roots."""
    xyz, q, camera = np.asarray(xyz, float), np.asarray(points, float), np.asarray(K, float)
    ids = sorted(set(map(int, anchor_ids)))
    if len(ids) < 3:
        return [], dict(reason="fewer_than_three_image_anchors", native_triple=[])
    triples = list(combinations(ids, 3))
    def spread(triple):
        image = q[list(triple)]
        objects = xyz[list(triple)]
        image_area = abs(float(np.cross(image[1]-image[0], image[2]-image[0])))
        object_area = float(np.linalg.norm(np.cross(objects[1]-objects[0], objects[2]-objects[0])))
        return image_area * object_area
    triple = max(triples, key=lambda value: (spread(value), tuple(-k for k in value)))
    if spread(triple) <= 1e-12:
        return [], dict(reason="degenerate_image_or_object_triangle", native_triple=list(triple))
    rays = np.column_stack([q[list(triple)], np.ones(3)]) @ np.linalg.inv(camera).T
    normalized = rays[:, :2]/rays[:, 2:]
    count, rotations, translations = cv2.solveP3P(
        np.ascontiguousarray(xyz[list(triple)]), np.ascontiguousarray(normalized),
        np.eye(3), None, flags=cv2.SOLVEPNP_P3P)
    result = [dict(name="P3P_ROOT_"+str(i), R=cv2.Rodrigues(rv)[0].tolist(),
                   t=np.asarray(tv).reshape(3).tolist(), used_ids=list(triple), estimated_dof=6,
                   support_mode="three_precise_image_anchors_with_finite_P3P_roots")
              for i, (rv, tv) in enumerate(zip(rotations, translations))]
    return result, dict(reason="finite_P3P_roots", native_triple=list(triple),
                        roots=int(count), full_pose_uniqueness_proved=False,
                        selection_used_ground_truth=False)


def _three_point_geometry(xyz, q, ids, R, t, K, cfg):
    ids = np.asarray(ids, int)
    image, objects = M._shape(q[ids]), M._shape(xyz[ids])
    jac = M._projection_jacobian(np.zeros(6), xyz[ids] @ np.asarray(R, float).T,
                                 K, np.asarray(t, float), 1.)
    jac /= np.maximum(np.linalg.norm(jac, axis=0), 1e-12)[None]
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-9))
    condition = float(singular[0]/max(singular[-1], 1e-15))
    usable = (rank == 6 and condition <= cfg.maximum_jacobian_condition
              and image["spread_ratio"] >= cfg.minimum_spread_ratio
              and image["singular_values"][1] >= cfg.minimum_spread_px
              and objects["spread_ratio"] > 1e-8)
    return dict(usable=bool(usable), reason="usable" if usable else "unstable_three_point_pose_geometry",
                jacobian_rank=rank, jacobian_condition=condition,
                image=image, object=objects, proves_global_uniqueness=False)


def select_region_pose(base_solution, xyz, K, initial_R, initial_t, regions,
                       proposals=(), config=None, *, image_sha256=None):
    """Score finite poses; protect image measurements and fixed anchor reference.

    Region penalties are empirical scores, not calibrated pose probabilities.
    A candidate must also pass the original visible-pool, depth, rank and motion
    guards. Coarse regions never increase the precise correspondence count.
    """
    from .keypoint_regions import evaluate_pose_regions
    from .protected_proposals import _context
    cfg = RegionCompletionConfig(**deepcopy(config or {}))
    if cfg.region_weight < 0 or not np.isfinite(cfg.region_weight) or cfg.maximum_roots != 4:
        raise ValueError("Finite nonnegative region weight and fixed four-root policy required")
    base = deepcopy(base_solution)
    (xyz, K, measured, R0, t0, _, _, pool, _, _, weights, acfg,
     pcfg, _, _) = _context(base, xyz, K, initial_R, initial_t)
    q = measured[:8]
    acfg = replace(acfg, anchor_protection=True)
    root_candidates, generation = p3p_candidates(xyz, q, K, base["anchor_ids"]) if cfg.include_p3p else ([], dict(reason="disabled"))
    candidates = [dict(name="BASE", R=base["R"], t=base["t"], used_ids=base["used_ids"],
                       estimated_dof=base["estimated_dof"], support_mode="unchanged_base_support")]
    for proposal in proposals:
        if (proposal.get("coordinate_frame") != "ORIGINAL_NATIVE_CF"
                or proposal.get("native_order_preserved") is not True
                or not np.allclose(np.asarray(proposal.get("cf_extents"), float), np.ptp(xyz, axis=0), atol=1e-12, rtol=0)):
            raise ValueError("Region pose proposals must preserve original native model and dimensions")
        candidate = deepcopy(proposal)
        candidate.update(used_ids=base["used_ids"], estimated_dof=base["estimated_dof"],
                         support_mode="unchanged_base_support")
        if base["estimated_dof"] == 3:
            candidate["R"] = R0.tolist()
        candidates.append(candidate)
    candidates += root_candidates[:cfg.maximum_roots]
    # Region evaluation only projects the supplied fixed model. No observations
    # are fabricated and no state or coordinate is changed by a hypothesis.
    assessment = evaluate_pose_regions(candidates, xyz, K, regions,
                                       raw_points=base["raw_points"], image_sha256=image_sha256)
    region_records = assessment["candidates"]
    initial_objective = M._aggregate_objective(xyz, q, K, R0, t0, R0, t0, pool, weights, pcfg)["objective"] if pool else 0.
    current_objective = M._aggregate_objective(xyz, q, K, np.asarray(base["R"]), np.asarray(base["t"]),
                                              R0, t0, pool, weights, pcfg)["objective"] if pool else 0.
    def region_penalty(record):
        return cfg.region_weight*pcfg.robust_px**2*record["normalized_penalty"]
    best, best_score = 0, current_objective + region_penalty(region_records[0])
    audits = []
    reference = base["anchor_reference"]
    if reference is None:
        reference = dict(R=R0.tolist(), t=t0.tolist(),
                         source="declared_initial_reference_without_accepted_seed")
    for i, (candidate, region) in enumerate(zip(candidates, region_records)):
        R, t = np.asarray(candidate["R"], float), np.asarray(candidate["t"], float)
        ids, dof = list(candidate["used_ids"]), int(candidate["estimated_dof"])
        if i == 0:
            audits.append(dict(name=candidate["name"], accepted_existing_base=True,
                               R=candidate["R"], t=candidate["t"],
                               objective=current_objective, region=region, combined_score=best_score))
            continue
        if len(ids) < 3 or not pool:
            audits.append(dict(name=candidate["name"], usable=False, reason="insufficient_precise_image_support", region=region))
            continue
        if (R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all()
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or abs(np.linalg.det(R)-1.) > 1e-6):
            audits.append(dict(name=candidate["name"], usable=False, reason="invalid_pose", region=region))
            continue
        geometry = (B._translation_geometry(xyz, q, K, R, t, ids, pcfg) if dof == 3 else
                    _three_point_geometry(xyz, q, ids, R, t, K, pcfg) if len(ids) == 3 else
                    M._geometry(xyz, q, ids, R, t, K, pcfg))
        camera = xyz @ R.T+t
        rotation = M._rotation_angle(R @ R0.T)
        translation = float(np.linalg.norm(t-t0)/max(np.linalg.norm(t0), 1e-6))
        shift = float(np.linalg.norm(M.project(xyz, R, t, K)-M.project(xyz, R0, t0, K), axis=1).max())
        motion_ok = rotation <= pcfg.maximum_rotation_deg and translation <= pcfg.maximum_translation_fraction and shift <= pcfg.maximum_projection_shift_px
        anchor = A._anchor_guard(xyz, q, K, R, t, base["anchor_ids"], reference, R0, t0, acfg, pcfg)
        objective = M._aggregate_objective(xyz, q, K, R, t, R0, t0, pool, weights, pcfg)["objective"]
        visible_ok = (objective <= current_objective+pcfg.validation_tolerance_px2
                      and objective <= initial_objective+pcfg.validation_tolerance_px2)
        score = objective + region_penalty(region)
        usable = bool(geometry["usable"] and (camera[:, 2] > 1e-6).all() and motion_ok
                      and anchor["usable"] and visible_ok and region.get("hard_valid", True))
        audits.append(dict(name=candidate["name"], usable=usable, precise_used_ids=ids,
                           R=R.tolist(), t=t.tolist(),
                           estimated_dof=dof, geometry=geometry, positive_depth=bool((camera[:, 2] > 1e-6).all()),
                           motion_pass=bool(motion_ok), anchor_guard=anchor, visible_objective_pass=bool(visible_ok),
                           objective=objective, region=region, combined_score=score,
                           support_mode=candidate["support_mode"]))
        if usable and score < best_score-pcfg.minimum_gain_px2:
            best, best_score = i, score
    selected = candidates[best]
    if best:
        R, t = np.asarray(selected["R"], float), np.asarray(selected["t"], float)
        used, dof = selected["used_ids"], selected["estimated_dof"]
        projected = M.project(xyz, R, t, K)
        completed = np.asarray(base["completed_points"], float)
        completed[:8] = projected
        base.update(R=R.tolist(), t=t.tolist(), R_cf=R.tolist(), centroid=t.tolist(),
                    accepted=True, used_ids=used, estimated_dof=dof, projected_points=projected.tolist(),
                    completed_points=completed.tolist(), final_rotation_exactly_initial=bool(np.array_equal(R, R0)),
                    reason="protected_region_ranked_finite_pose", fallback_reason=None,
                    final_anchor_guard=audits[best]["anchor_guard"],
                    uncertainty=A._uncertainty(xyz, K, R, t, t0, used, weights, dof, pcfg))
        base["final_supplemental_ids"] = [k for k in used if k not in base["anchor_ids"]]
        base["omitted_pool_ids"] = [k for k in pool if k not in used]
        base["base_anchor_reference"] = deepcopy(base["anchor_reference"])
        base["anchor_reference"] = deepcopy(reference)
        base["final_guard"] = dict(policy="region_candidate_visible_tolerance", source_V1_guard=base["final_guard"],
                                  initial_objective=initial_objective, selected_objective=audits[best]["objective"],
                                  selected_checkpoint_passes_full_pool_guard=bool(initial_objective-audits[best]["objective"] > pcfg.minimum_gain_px2),
                                  selected_checkpoint_passes_visible_tolerance_guard=True,
                                  strict_initial_objective_gain=initial_objective-audits[best]["objective"],
                                  tolerance_px2=pcfg.validation_tolerance_px2,
                                  independent_boundary_seed_exception=False)
        for row in base["keypoint_status"]:
            row["used_for_final_pose"] = row["corner"] in used
        for key, row in base["observed_measurements_by_id"].items():
            row["used_for_final_pose"] = int(key) in used
    base.update(region_audit=dict(generation=generation, candidates=audits, selected=selected["name"],
                                 region_weight=cfg.region_weight, region_count=len(regions),
                                 regions_are_precise_correspondences=False, calibrated_probability=False,
                                 finite_candidates_contain_true_pose_guaranteed=False,
                                 selected_reference_source=reference["source"],
                                 rejected_addition_and_deferred_ids_are_historical_V1=True,
                                 original_three_point_V1_policy_changed_only_in_this_opt_in_solver=True),
                selected_pose_source=selected["name"], rigid_model_corners=base["projected_points"])
    return base
