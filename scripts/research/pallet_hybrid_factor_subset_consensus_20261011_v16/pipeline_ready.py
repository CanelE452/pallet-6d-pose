"""Two fixed solver ablations on the same sealed V7 cornerwise hybrid supply.

No model/decoder/LOO corner selection/initial pose/file/GT read is performed
here. The driver owns full byte, population, cleanup and resource proofs.
Original N3 coordinates are explicitly independent RGB observations in this
separate hybrid experiment; the V15 boundary-only path remains immutable.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np

from ..pallet_same_observation_controls_20261010_v8 import controls as V8
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import assemble
from ..pallet_cornerwise_independent_20261010_v4 import selection as S4
from ..pallet_factor_subset_consensus_20261011_v15 import pipeline as P15
from ..pallet_factor_subset_consensus_20261011_v15.solver import FactorSubsetConsensusBank
from ..pallet_cornerwise_independent_20261010_v4.pose import PoseBank

PARENT_METHOD='IMAGE_ROLE_CORNERWISE_HYBRID'
METHODS=('HYBRID_FACTOR_SUBSET_FINITE_EDGE_CONSENSUS','HYBRID_FACTOR_SUBSET_POINT_ONLY')
PRIMARY=METHODS[0]
POLICY=dict(schema='sealed_cornerwise_hybrid_factor_subset_ablation_v16',
    methods=list(METHODS),primary=PRIMARY,parent_method=PARENT_METHOD,
    observation_supply='exact sealed V7 IMAGE_ROLE_CORNERWISE_HYBRID',
    corner_selection_policy=copy.deepcopy(S4.POLICY),corner_selection_reexecuted=False,
    native_N3_is_explicit_independent_RGB_observation=True,
    boundary_only_missing_queries_are_not_filled=True,
    solver=copy.deepcopy(P15.POLICY['solver']),
    actual_numeric_solve_calls_per_frame=2,shared_same_coordinate_four_ID_hypotheses=True,
    point_only_has_no_line_factors=True,
    point_line_consumes_only_actually_adopted_boundary_source_edges=True,
    endpoint_decisions='fresh unchanged native endpoint LOO on resulting unused line pool',
    endpoint_unknown_retained=True,endpoint_contradicted_only_removed=True,
    original_H_unchanged=True,initial_pose_start_only=True,center_and_H_xy_fit=False,
    NEW_H_projection_once_without_refit=True,
    training_updates=0,new_RGB=0,new_model_forwards=0,new_corner_selection_calls=0,
    threshold_or_weight_search=False,known_DEV=True,unseen_evaluation=False,latency_claim=False)


def require(v,s): V8.require(v,s)
def clone(v): return copy.deepcopy(v)
def semantic(v): return V8.semantic_sha256(v)


def validate_hybrid_packet(packet):
    """Structural checks; file-byte/full-population authenticity is external."""
    require(isinstance(packet,Mapping) and set(packet)=={
        'parent_row','native_N3_points','parent_completion','role_observation'},
        'explicit GT-free hybrid parent packet required')
    row,done,obs=packet['parent_row'],packet['parent_completion'],packet['role_observation']
    require(isinstance(row,Mapping) and isinstance(done,Mapping) and isinstance(obs,Mapping),
        'parent, completion and original observation mappings required')
    require(done.get('complete') is True and done.get('frames')==245 and
        done.get('geometry_rows')==1960 and done.get('observation_rows')==735 and
        done.get('fixed_rows')==490 and done.get('cleanup_error','ABSENT') is None,
        'incomplete or unclean V7 parent population')
    for key in ('protocol_sha256','geometry_sha256','inference_receipt_sha256'):
        V8._sha(done.get(key),'parent '+key)
    require(not set(row)&P15.V11.A10.TRUTH_FIELDS and not set(obs)&P15.V11.A10.TRUTH_FIELDS,
        'scored or truth fields cannot enter hybrid observations')
    require(row.get('method')==PARENT_METHOD and row.get('head_arm')==row.get('head_mode')=='IMAGE_ROLE'
        and row.get('observation_supply')=='CORNERWISE_HYBRID'
        and row.get('head_checkpoint_sha256')==V8.ROLE_CHECKPOINT_SHA256,
        'unchanged corrected V7 IMAGE_ROLE hybrid parent required')
    require(row.get('oracle') is False and row.get('partial_lines_not_pose_inputs') is True
        and row.get('final_numeric_pose_has_initial_prior') is False,
        'parent is not the sealed nonoracle point-only hybrid control')
    require(isinstance(row.get('id'),str) and isinstance(row.get('session'),str)
        and isinstance(row.get('fixed_metadata'),Mapping)
        and row['fixed_metadata'].get('preserved') is True,'parent identity or metadata absent')
    require(isinstance(row.get('calibration_binding'),Mapping),'original calibration binding absent')
    V8._sha(row['calibration_binding'].get('sha256'),'original calibration')
    q,native,K,xyz=(np.asarray(v,float) for v in (row.get('input_points'),packet['native_N3_points'],
        row.get('K'),row.get('xyz')))
    require(q.shape==native.shape==(9,2) and not np.isinf(q).any() and not np.isinf(native).any(),
        'selected/native coordinates must be 9x2 finite or missing')
    require(K.shape==(3,3) and np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0
        and xyz.shape==(3,) and np.isfinite(xyz).all() and (xyz>0).all(),'invalid K or physical XYZ')
    hw=row.get('raw_hw');require(isinstance(hw,(list,tuple)) and len(hw)==2
        and all(isinstance(v,int) and not isinstance(v,bool) and v>0 for v in hw),'invalid image size')
    H=V8._ids(row.get('hidden_initial'),'original H')
    require(H==V8._ids(row.get('predicted_initial_N3_hidden'),'predicted initial H'),'H declarations differ')
    contract=row.get('observation_contract')
    require(isinstance(contract,Mapping) and contract.get('schema')=='actual_observation_only_cornerwise_contract_v4'
        and contract.get('cornerwise_policy')==S4.POLICY,'frozen V4 corner selection contract differs')
    adopted=V8._ids(contract.get('hybrid_boundary_corner_ids'),'adopted boundary IDs')
    require(adopted==V8._ids(row.get('selected_corner_ids'),'selected corner IDs')
        and set(adopted).isdisjoint(H),'selected corner source/H differs')
    require(contract.get('native_N3_RGB_corner_ids')==[k for k in range(8) if k not in adopted]
        and contract.get('actual_self_hidden_ids')==H
        and contract.get('same_N3_bank_reused_across_heldouts') is True
        and contract.get('heldout_projection_is_observation') is False
        and contract.get('LOO_initial_prior_includes_heldout_influence') is False,
        'native/LOO source contract differs')
    require(obs.get('id')==row['id'] and obs.get('session')==row['session']
        and obs.get('head_arm')=='IMAGE_ROLE' and obs.get('GT_input') is False,
        'same-frame GT-free original IMAGE_ROLE observation required')
    require(semantic(obs.get('initial_N3_pose'))==semantic(row.get('initial_pose'))
        and obs.get('predicted_N3_hidden')==H
        and np.array_equal(np.asarray(obs.get('native_N3_points'),float),native,equal_nan=True),
        'original initial pose/H/native identity differs')
    require(obs.get('raw_logits_sha256')==row.get('observation_raw_logits_sha256'),
        'original ROLE logit identity differs')
    require(all(isinstance(obs.get(k),list) for k in ('corners','lines','queries')),
        'full original decoded ROLE observation required')
    count=0 if row.get('selected_index') is None and obs.get('no_detection') is True else 84
    require([v.get('query') for v in obs['queries']]==list(range(count)),'query IDs/order differ')
    for line in obs['lines']:
        edge,queries=line.get('edge'),line.get('queries')
        require(isinstance(edge,int) and not isinstance(edge,bool) and 0<=edge<12
            and isinstance(queries,list) and all(isinstance(i,int) and not isinstance(i,bool)
            and 0<=i<84 and i//7==edge for i in queries),'source line/query identity differs')
        require(all(obs['queries'][i].get('calibrated_model_coverage') is True
            and obs['queries'][i].get('selected_xy') is not None for i in queries)
            and np.array_equal(np.asarray(line.get('support_points'),float),
            np.asarray([obs['queries'][i]['selected_xy'] for i in queries],float)),
            'line does not contain exact original source-CAL query observations')
    corners={int(c['id']):c for c in obs['corners']}
    records={int(c['id']):c for c in contract.get('cornerwise_records',[])}
    require(len(corners)==len(obs['corners']) and len(records)==len(contract.get('cornerwise_records',[])),
        'duplicate original corner or LOO decision IDs')
    for k in range(8):
        if k in adopted:
            require(k in corners and k in records and records[k].get('accepted') is True
                and np.array_equal(q[k],np.asarray(corners[k]['xy'],float))
                and np.array_equal(q[k],np.asarray(records[k]['candidate_xy'],float)),
                'adopted coordinate is not original decoded boundary with frozen accepted LOO')
            require(records[k].get('heldout_fit_requested_excluded_ids')==sorted(set(H)|{k})
                and records[k].get('fit_exclusion_verified') is True
                and records[k].get('projected_coordinate_used_as_observation') is False,
                'frozen corner LOO exclusion/projection contract differs')
        else:
            require(np.array_equal(q[k],native[k],equal_nan=True),'native RGB observation changed or fabricated')
    require(np.array_equal(q[8],native[8],equal_nan=True),'detector center changed')
    require(isinstance(row.get('initial_pose'),Mapping) and isinstance(row.get('solver'),Mapping)
        and row['solver'].get('prior_used') is False
        and row['solver'].get('known_dimension_constraint_used') is False
        and row['solver'].get('hidden')==H,'parent solver prior/H differs')
    return dict(row=row,completion=done,observation=obs,q=q.copy(),native=native.copy(),K=K.copy(),
        xyz=xyz.copy(),H=H,size=(hw[1],hw[0]),contract=clone(contract),adopted=adopted)


def solve_hybrid_packet(packet,*,observer=None):
    """Two real solves; shared raw4 hypotheses, one fresh endpoint audit."""
    p=validate_hybrid_packet(packet);before=semantic(packet);parent=p['row']
    bank=PoseBank(p['q'],p['K'],p['xyz'],image_size=p['size']);bank_before=clone(bank.ledger)
    views,line_ledger=P15.repair_observations(p['q'],p['native'],p['K'],p['xyz'],p['observation'],p['H'],
        p['adopted'],initial_pose=clone(parent['initial_pose']),point_result=clone(parent['solver']),
        image_size=p['size'],endpoint_audit=None,observer=observer,point_bank=bank)
    line=views[P15.PRIMARY];after_line=clone(bank.ledger)
    # Explicit no-line ablation. The same selected q remains unchanged; no
    # source edge can be counted or consumed when all line factors are absent.
    point_observation=clone(p['observation']);point_observation['lines']=[]
    point_bank=FactorSubsetConsensusBank(p['q'],p['K'],p['xyz'],image_size=p['size'],point_bank=bank)
    point=point_bank.solve(point_observation,adopted_boundary_corner_ids=(),hidden=p['H'],
        initial_pose=clone(parent['initial_pose']),point_result=clone(parent['solver']))
    after_point=clone(bank.ledger)
    require(point['line_edges']==point['fit_line_edges']==[] and point['consumed_edges']==[],
        'line factor entered point-only ablation')
    require(point['hypothesis_bank_operation_counts']['generic_cache_misses']==0
        and point['hypothesis_bank_operation_counts']['generic_calls']==0,
        'point-only repeated the same raw4 numerical hypothesis')
    rows=[]
    for method,solved in zip(METHODS,(line,point)):
        require(set(p['H']).isdisjoint(solved['fit_input_ids']) and solved['prior_used'] is False
            and solved['known_dimension_constraint_used'] is False,'H or dimension prior entered fit')
        row={k:clone(parent[k]) for k in ('id','session','K','xyz','raw_hw','selected_index','fixed_metadata',
            'head_arm','head_mode','head_checkpoint_sha256','calibration_binding')}
        row.update(assemble(p['native'],p['q'],clone(parent['initial_pose']),p['H'],solved,
            'N3_VALIDATED_ROLE',p['contract']))
        row.update(method=method,parent_method=PARENT_METHOD,observation_supply='CORNERWISE_HYBRID',
            selected_corner_ids=list(p['adopted']),original_native_N3_points=p['native'].tolist(),
            predicted_initial_N3_hidden=list(p['H']),initial_H_applied=True,hybrid_consensus_policy=clone(POLICY),
            original_self_hidden_initial=list(p['H']),cornerwise_selection=clone(p['contract']['cornerwise_selection']),
            observation_raw_logits_sha256=parent.get('observation_raw_logits_sha256'),
            observation_filter=clone(line_ledger['observation_filter']) if method==PRIMARY else None,
            native_N3_is_numeric_final_pose_observation=True,missing_sparse_filled_for_numeric_fit=False,
            partial_lines_not_pose_inputs=method!=PRIMARY,local_point_line_refinement=method==PRIMARY,
            pose_estimation_kind='FIXED_HYBRID_FACTOR_SUBSET_CONSENSUS',
            initial_pose_may_supply_local_optimizer_start=method==PRIMARY,
            numerical_NEW_is_not_pose_accuracy_success=True,global_uniqueness_proven=False,
            parent_geometry_row_semantic_sha256=semantic(parent),parent_observation_row_semantic_sha256=semantic(p['observation']),
            parent_native_N3_semantic_sha256=semantic(p['native']),parent_completion=clone(p['completion']),
            point_only_observation_semantic_sha256=semantic(point_observation),
            head_forward_calls=0,calibration_or_decode_calls=0,new_corner_selection_calls=0,
            cached_accuracy_row_used=False,solver_replay_latency_claim=False,oracle=False)
        require(np.array_equal(np.asarray(row['input_points']),p['q'],equal_nan=True)
            and np.array_equal(np.asarray(row['native_points'])[8],p['native'][8],equal_nan=True),
            'selected observations or center changed')
        require(row['reprojections_reused_as_observations'] is False,'hidden projection was re-fit')
        rows.append(row)
    require(before==semantic(packet),'immutable hybrid parent packet changed')
    return rows,dict(schema='same_hybrid_factor_subset_paired_ledger_v16',id=parent['id'],session=parent['session'],
        actual_filtered_solver_calls=2,reported_output_views=2,methods=list(METHODS),
        input_packet_semantic_sha256=before,parent_geometry_row_semantic_sha256=semantic(parent),
        parent_observation_row_semantic_sha256=semantic(p['observation']),
        parent_native_N3_semantic_sha256=semantic(p['native']),selected_input_semantic_sha256=semantic(p['q']),
        actual_adopted_boundary_corner_ids=list(p['adopted']),initial_H=list(p['H']),
        native_RGB_corner_ids=[k for k in range(8) if k not in p['adopted']],
        same_selected_coordinate_bank_shared=True,shared_point_bank_input_hash=bank.digest,
        shared_point_bank_before=bank_before,shared_point_bank_after_line=after_line,shared_point_bank_after_point=after_point,
        shared_point_bank_operation_counts={k:bank.ledger[k]-bank_before[k] for k in bank.ledger},
        point_line_ledger=line_ledger,point_only_solver=clone(point),point_only_bank_counts=clone(point_bank.counts),
        point_only_hypothesis_bank_operation_counts=clone(point['hypothesis_bank_operation_counts']),
        source_geometry_ownership_truth='UNKNOWN',copied_V7_corner_selection_not_new_LOO=True,
        new_GT_scores_models_training_RGB_calls=0,latency_claim=False)
