"""Pure two-family cuboid fitting from the same frozen image observations.

The image-facing native corner channels need not identify which registered
horizontal dimension is width. Both W/D families are therefore fitted. The
second family uses a proper basis change with the same unordered physical
cuboid and the same initial rendered box. No observation/GT/visibility gate
is re-estimated here. Every S remains excluded; all O/T and U regions remain
simultaneous constraints. The coarse prior weight is explicitly zero.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from . import common as C
from . import joint_ot_pnp as V3
from . import statewise_partial_boundary_u_solver_v4_6 as PARENT

RY90 = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])


def _unordered_gap(first, second):
    distance = np.linalg.norm(np.asarray(first)[:,None]-np.asarray(second)[None],axis=2)
    return float(max(distance.min(axis=0).max(),distance.min(axis=1).max()))


def make_wd_families(xyz, coarse_pose, initializations=None):
    """Declare both bases; this helper reads geometry only, never raw points."""
    xyz = V3._finite_array(xyz,(8,3),"original native cuboid")
    extents = np.ptp(xyz,axis=0)
    if np.any(extents<=0) or not np.allclose(xyz,C.cuboid(*extents),atol=1e-9,rtol=0):
        raise ValueError("Original ordered centered registered cuboid required")
    R = V3._finite_array(coarse_pose["R"],(3,3),"original coarse rotation")
    t = V3._finite_array(coarse_pose["t"],(3,),"original coarse translation")
    Q = V3._finite_array(coarse_pose.get("physical_registration_Q",np.eye(3)),(3,3),"original physical registration")
    if not V3._valid_rotation(R) or not V3._valid_rotation(Q):
        raise ValueError("Coarse R and registration Q must be proper rotations")
    if "R_physical" in coarse_pose and not np.allclose(R@Q,coarse_pose["R_physical"],atol=1e-7,rtol=0):
        raise ValueError("Original physical rotation binding mismatch")
    supplied = list(initializations or [])
    original_coarse = deepcopy(dict(coarse_pose))
    original_coarse.update(R=R.tolist(),t=t.tolist(),physical_registration_Q=Q.tolist())
    if "R_physical" in coarse_pose:
        original_coarse["R_physical"]=(R@Q).tolist()
    families = [dict(name="ORIGINAL_WD",native_xyz=xyz.tolist(),cf_extents=extents.tolist(),
        physical_registration_Q=Q.tolist(),coarse_pose=original_coarse,
        initializations=None if initializations is None else deepcopy(supplied),
        initial_rendered_vertex_set_gap_m=0.,registered_physical_vertex_set_gap_m=0.,
        physical_initial_rotation_gap=0.,extent_diagonal_m=float(np.linalg.norm(extents)))]
    if np.isclose(extents[0],extents[2],atol=1e-9,rtol=0):
        return families
    swapped_extents = extents[[2,1,0]]
    swapped_xyz = C.cuboid(*swapped_extents)
    swapped_Q = RY90@Q
    swapped_R = R@RY90.T
    swapped_coarse = deepcopy(original_coarse)
    swapped_coarse.update(R=swapped_R.tolist(),physical_registration_Q=swapped_Q.tolist())
    if "R_physical" in coarse_pose:
        swapped_coarse["R_physical"]=(swapped_R@swapped_Q).tolist()
    swapped_initializations = []
    for start in supplied:
        value = deepcopy(dict(start))
        start_R = V3._finite_array(start["R"],(3,3),"supplied original-basis rotation")
        if not V3._valid_rotation(start_R):
            raise ValueError("Supplied initialization must be a proper rotation")
        value["R"] = (start_R@RY90.T).tolist()
        swapped_initializations.append(value)
    render_gap = _unordered_gap(xyz@R.T+t,swapped_xyz@swapped_R.T+t)
    physical_gap = _unordered_gap(xyz@Q,swapped_xyz@swapped_Q)
    rotation_gap = float(np.linalg.norm(R@Q-swapped_R@swapped_Q))
    if max(render_gap,physical_gap,rotation_gap)>1e-9:
        raise RuntimeError("W/D basis change altered the known physical cuboid or initial rendered shape")
    families.append(dict(name="SWAPPED_WD",native_xyz=swapped_xyz.tolist(),cf_extents=swapped_extents.tolist(),
        physical_registration_Q=swapped_Q.tolist(),coarse_pose=swapped_coarse,
        initializations=None if initializations is None else swapped_initializations,
        initial_rendered_vertex_set_gap_m=render_gap,
        registered_physical_vertex_set_gap_m=physical_gap,physical_initial_rotation_gap=rotation_gap,
        extent_diagonal_m=float(np.linalg.norm(swapped_extents))))
    return families


def _physical_rotation_difference(first, second):
    """Registered rectangular-pallet C2 equivalence; no reference pose used."""
    first,second=np.asarray(first),np.asarray(second)
    return min(float(np.rad2deg(Rotation.from_matrix(first@(second@sym).T).magnitude()))
               for sym in (np.eye(3),np.diag([-1.,1.,-1.])))


def solve_two_wd_families(xyz: Any,K: Any,states: Sequence,visible: Any,domains: Any,
                          image_hw: Sequence,coarse_pose: Mapping,
                          initializations: Any=None,config: Mapping|None=None,*,
                          boundary_observations: Any=None,u_regions: Any=None,
                          solver_callable: Callable|None=None) -> dict:
    """Fit both complete hypotheses and retain every candidate family solution.

    Selection is minimum unchanged image likelihood among joint-feasible
    families. Exact ties retain ORIGINAL_WD. The initial S gate, source q and
    covariances, O/T/U localization records, partial lines and physical size
    never change. ``solver_callable`` is an explicit optional pure evaluator;
    its binding must be recorded by the caller. No global monkeypatch is used.
    """
    source_visible=deepcopy(V3._records(visible))
    source_domains=deepcopy(V3._records(domains))
    source_states=[V3._state(s) for s in states]
    if len(source_states)!=8:
        raise ValueError("Eight authoritative frozen states required")
    configuration=PARENT._configuration(config)
    requested_prior_weight=configuration["prior_weight"]
    configuration["prior_weight"]=0.
    models=make_wd_families(xyz,coarse_pose,initializations)
    solver=PARENT.solve_boundary_u_region_joint_ot_pnp if solver_callable is None else solver_callable
    records=[]
    for model in models:
        solution=solver(model["native_xyz"],K,deepcopy(source_states),deepcopy(source_visible),
            deepcopy(source_domains),image_hw,deepcopy(model["coarse_pose"]),
            initializations=deepcopy(model["initializations"]),config=deepcopy(configuration),
            boundary_observations=deepcopy(boundary_observations),u_regions=deepcopy(u_regions))
        if solution["accepted"]:
            if not np.isfinite(solution.get("objective",np.nan)):
                raise RuntimeError("Accepted family must have a finite unchanged image objective")
            if not np.allclose(np.asarray(solution["R"])@np.asarray(model["physical_registration_Q"]),
                               solution["R_physical"],atol=1e-7,rtol=0):
                raise RuntimeError("Family solver physical rotation binding mismatch")
        records.append({**deepcopy(model),"accepted":bool(solution["accepted"]),
            "objective":float(solution["objective"]) if solution["accepted"] else None,
            "solution":solution})
    eligible=[(i,r) for i,r in enumerate(records) if r["accepted"]]
    selected_index=min(eligible,key=lambda item:(item[1]["objective"],item[0]))[0] if eligible else 0
    selected=records[selected_index]
    selected_solution=deepcopy(selected["solution"])
    alternatives=[]
    if eligible:
        for i,record in eligible:
            if i==selected_index:
                continue
            gap=record["objective"]-selected["objective"]
            rot=_physical_rotation_difference(selected_solution["R_physical"],record["solution"]["R_physical"])
            trans=float(np.linalg.norm(np.asarray(selected_solution["t"])-record["solution"]["t"])
                        /selected["extent_diagonal_m"])
            material=(rot>configuration["ambiguity_rotation_deg"]
                      or trans>configuration["ambiguity_translation_fraction"])
            alternatives.append(dict(family=record["name"],objective_gap=gap,
                physical_rotation_difference_C2_deg=rot,translation_difference_fraction=trans,
                materially_distinct=material,near_equal_image_objective=gap<=configuration["ambiguity_objective_gap"]))
    cross_ambiguous=any(a["materially_distinct"] and a["near_equal_image_objective"] for a in alternatives)
    if cross_ambiguous and selected_solution["accepted"]:
        # Other-family poses use another explicit native basis. Preserve them
        # in separate family records, never mix them into same-basis samples.
        selected_solution["within_selected_family_ambiguity"]=deepcopy(selected_solution.get("ambiguity"))
        selected_solution.setdefault("ambiguity",{})["detected"]=True
        selected_solution["ambiguity"]["cross_WD_family_ambiguity"]=True
        selected_solution["ambiguity"]["global_uniqueness_proved"]=False
        selected_solution["status"]="joint_feasible_WD_family_ambiguous"
        selected_solution["OT_solved"]=False
        selected_solution["OT_unique_solution_claimed"]=False
    return dict(schema="STATEWISE_TWO_WD_FAMILY_V4_9",accepted=selected["accepted"],
        selected_family=selected["name"],selected_family_index=selected_index,solution=selected_solution,
        selected_model_geometry={k:deepcopy(selected[k]) for k in ("native_xyz","cf_extents","physical_registration_Q")},
        selected_coarse_pose=deepcopy(selected["coarse_pose"]),families=records,
        family_selection=dict(rule="minimum same soft BI/partial image objective among joint-feasible families",
            exact_tie_rule="ORIGINAL_WD first",GT_inputs=False,image_reads=0,
            mask_selector_used=False,all_joint_hypotheses_searched=True),
        family_ambiguity=dict(detected=cross_ambiguous,alternatives=alternatives,
            globally_unique_family_proved=False,physical_symmetry_assumption="registered rectangular pallet C2"),
        solver_configuration=configuration,requested_prior_weight=requested_prior_weight,
        effective_prior_weight=0.,original_states=source_states,states_preserved=True,
        original_BI_xy_covariance_preserved=True,all_OT_U_domains_preserved=True,
        all_OT_U_and_partial_B_simultaneous=True,original_S_gate_preserved=True,S_image_queries=0,
        new_RGB_or_state_forwards=0,GT_inputs=False,new_covariance_or_prior_fit=False,
        native_channel_ids_not_permuted=True,physical_cuboid_size_preserved=True,
        registration_change_explicit=True,original_physical_registration_Q=deepcopy(models[0]["physical_registration_Q"]))

