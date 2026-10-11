"""Independent geometry checks; no PnP, training, protected frames or GT routing."""
import argparse
import math
from pathlib import Path
from collections import Counter
import numpy as np
from . import common as C
from . import geometry_analysis as A
from . import geometry_perturbations as P


def main():
    argparse.ArgumentParser().parse_args()
    ids=A.read(A.DOC/'A3_GEOMETRY_DIAGNOSTIC.json')['frame_ids']
    gg=P.geometry(C.SOURCE,ids)
    historical={r['id']:r for r in A.rows(C.PUBLIC/A.WD_REL/'STAGE1_ROWS_SYNTH.jsonl.gz')
                if r['seed']==1 and r['method']=='N3_THEN_SUBPIX' and r['id'] in ids}
    pose=A.pose_api(C.SOURCE);roundtrip=[]
    for fid in ids:
        r=historical[fid];g=gg[fid];xyz=np.asarray(g['dims']);name=r['hypOracle']
        dims=np.array([max(xyz[0],xyz[2]),xyz[1],min(xyz[0],xyz[2])] if name==A.NAMES[0]
                      else [min(xyz[0],xyz[2]),xyz[1],max(xyz[0],xyz[2])])
        model=pose.cuboid(*dims)
        S=np.linalg.lstsq(model,g['Xcf'],rcond=None)[0].T
        Q=np.eye(3) if abs(dims[0]-xyz[0])<1e-6 else pose.rotations(4)[1]
        residual_basis=S@Q@np.diag([1.,-1.,-1.])
        error=float(np.max(abs(model@S.T-g['Xcf'])))
        orth_error=float(np.max(abs(S.T@S-np.eye(3))))
        assert error<1e-12 and orth_error<1e-12 and abs(np.linalg.det(S)-1)<1e-12
        angles=[float(np.degrees(np.arccos(np.clip((np.trace(q.T@residual_basis)-1)/2,-1,1))))
                for q in pose.rotations(r['pose_symmetry_order'])]
        width=np.linalg.norm(g['Xcf'][1]-g['Xcf'][0]);depth=np.linalg.norm(g['Xcf'][4]-g['Xcf'][0])
        label=A.NAMES[0] if width>depth else A.NAMES[1]
        assert label==name
        roundtrip.append(dict(id=fid,group_order=r['pose_symmetry_order'],
            ordered_model_rigid_map_max_error_m=error,rigid_map_orthogonality_error=orth_error,
            rigid_map_determinant=float(np.linalg.det(S)),
            fixed_physical_restore_remaining_rotation=np.round(residual_basis,12).tolist(),
            physical_rotation_error_mod_approved_group_deg=min(angles),
            physical_roundtrip_pass=min(angles)<1e-4,
            long_short_edge_label_roundtrip_pass=label==name))
    failures=[r['id'] for r in roundtrip if not r['physical_roundtrip_pass']]
    assert set(failures)=={'G38__G__f30814','G38__G__f30131'}
    A.write('A3_PHYSICAL_ROUNDTRIP_AUDIT.json',dict(status='IMPLEMENTATION_OR_CONTRACT_ISSUE',
        long_short_parity_label_status='PASS_16_OF_16',physical_pose_roundtrip_status='FAIL_2_OF_16',
        failure_ids=failures,rows=roundtrip,
        algebra='Xcf = Mcf S^T; exact Rcf = Rrenderer S; frozen F restores Rphysical=Rcf Q Rx(pi); hence residual physical basis B=S Q Rx(pi)',
        finding='For two C1 frames B is yaw180, which is not in the approved C1 group; camera-facing coordinates omit the canonical front/back signed branch.',
        affected_interpretation='A3 full-R/success noise curve cannot establish a detector-noise-only geometric limit. Raw results retained with this gate failure.',
        unaffected='Long/short labels depend on ordered edge01 vs edge04 lengths; yaw180 preserves lengths, so label mapping passes.',
        no_GT_based_pose_repair=True,no_approved_symmetry_changed=True,
        new_PnP_calls=0,new_protected_processing=0))
    allowed=set(A.read(A.DOC/'INPUT_CONTRACT.json')['allowed_ids'])
    candidates=list(A.rows(A.DOC/'AE_CANDIDATE_METRICS.jsonl.gz'))
    assert len(candidates)==173*24 and {r['id'] for r in candidates}==allowed
    assert len({(r['backbone'],r['method'],r['seed'],r['id']) for r in candidates})==len(candidates)
    assert all(set(r['candidate_metrics'])==set(A.NAMES) for r in candidates)
    assert len([r for r in candidates if r['axis_reference_status']=='UNCLEAR'])==3*24
    oracle=list(A.rows(A.DOC/'A2_ORACLE_ROWS.jsonl.gz'))
    assert len(oracle)==len(candidates)*5
    maximum_error=0.;checks=0
    aggregate=A.read(A.DOC/'A2_ORACLE_METRICS.json')['summaries']
    for summary in aggregate:
        matching=[r for r in oracle if (r['population'],r['backbone'],r['method'],r['arm'])==
                  (summary['population'],summary['backbone'],summary['method'],summary['arm'])]
        for key in ('translation_cm','rotation_deg','success','available','axis_wrong','wd_confusion'):
            per={}
            for r in matching:
                if r.get(key) is not None:per.setdefault(r['id'],[]).append(r[key])
            values=[math.fsum(rr)/len(rr) for rr in per.values()]
            mean=math.fsum(values)/len(values) if values else None
            actual=summary['metrics'][key]['mean']
            assert (mean is None)==(actual is None)
            if mean is not None:
                maximum_error=max(maximum_error,abs(mean-actual));assert abs(mean-actual)<1e-9
            checks+=1
    for curve in A.read(A.DOC/'E_RISK_COVERAGE.json')['curves']:
        assert curve['frames']==173
        for point in curve['curve']:
            assert point['accepted']+point['abstained']==173
            assert abs(point['coverage']-point['accepted']/173)<1e-12
            assert 0<=point['accepted_success_over_all']<=point['coverage']
    zeros=[r for r in A.rows(A.DOC/'G2_PLANE_SENSITIVITY_ROWS.jsonl.gz') if r['delta']==0]
    assert zeros and all(not r['changed_from_exact_plane'] for r in zeros)
    original=A.read(A.DOC/'GEOMETRY_CODE_LOCK.json')
    for binding in original['source_bindings']:
        base=C.PUBLIC if binding['owner']=='published_source' else C.SOURCE
        assert C.sha(base/binding['path'])==binding['sha256']
    original_code=original['code_bindings'][0]
    assert C.sha(C.ROOT/original_code['path'])==original_code['sha256']
    final=A.read(A.DOC/'GEOMETRY_EXECUTION_AMENDMENT.json')['final_code_binding']
    assert C.sha(C.ROOT/final['path'])==final['sha256']
    A.write('GEOMETRY_VERIFICATION.json',dict(status='PASS_NUMERICS_AND_DENOMINATORS_PHYSICAL_A3_GATE_FAIL_RETAINED',
        independent_math_fsum_mean_checks=checks,maximum_mean_difference=maximum_error,
        candidate_rows=len(candidates),oracle_rows=len(oracle),permitted_frames=173,
        protected_newly_scored=0,original_core_hash_unchanged=True,
        G2_zero_perturbation_unchanged_rows=len(zeros),
        references_never_predictor_input=True,raw_reference_coordinates_published=False,
        physical_A3_gate='FAIL signed C1 restoration; see A3_PHYSICAL_ROUNDTRIP_AUDIT.json',
        long_short_edge_roundtrip='PASS16',new_PnP_calls=0))
    print('GEOMETRY VERIFICATION',checks,'max_mean_difference',maximum_error,'C1_roundtrip_failure_ids',failures,flush=True)


if __name__=='__main__':main()
