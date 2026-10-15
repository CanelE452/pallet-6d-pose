"""Independent geometry/filter/current-guard audit of the sealed U control."""
from pathlib import Path
from collections import Counter
import argparse
import numpy as np

from . import evaluate_grazing_native_candidates_v4_18 as V


def main(output=V.OUTPUT):
    output=Path(output);dest=output/'independent_root_audit_v1'
    if dest.exists():raise FileExistsError('Preserve prior audit')
    C,B=V.C,V.B;protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREDICTION_SEAL.json')
    if seal['rows']!=238 or C.sha(output/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']:
        raise ValueError('Complete238 prediction seal required')
    for key in('source_sha256','input_sha256','protected_user_sha256'):B.E._verify_sources(protocol[key])
    original={r['id']:r for r in C.rows(B.OUTPUT/'INPUTS.jsonl.gz')}
    adapted={r['id']:r for r in C.rows(output/'INPUTS.jsonl.gz')}
    rows=list(C.rows(output/'PREDICTIONS.jsonl.gz'));angle=B.ANGLE.authentication_angular_parameters(B._verified_model())
    checks=0;failures=[];maximum=0.;total=0
    def check(ok,what,ident=None,arm=None):
        nonlocal checks
        checks+=1
        if not bool(ok):failures.append(dict(check=what,id=ident,arm=arm))
    check([r['id']for r in rows]==protocol['ids'],'ordered238')
    check(Counter(r['partition']for r in rows)==Counter(CAL=46,TEST=49,REMAINDER=143),'46+49+143 severe0')
    for row in rows:
        task=adapted[row['id']];old=original[row['id']];ident=row['id'];family=task['families']['ORIGINAL_WD']
        geom=dict(native_xyz=family['native_xyz'],R_cf=family['coarse_pose']['R'],centroid=family['coarse_pose']['t'],
            initial_self_hidden=task['initial_self_hidden'])
        gate=V.G.apply_grazing_BI_uncertainty(geom,old['states'],geometry_tolerance_deg=task['spec']['config']['geometry_tolerance_deg'])
        total+=len(gate['new_U_native_ids'])
        check(gate['states']==task['states']==row['states'],'original geometry gate recomputed',ident)
        check(gate['new_U_native_ids']==task['grazing_adaptation']['geometry_gate']['new_U_native_ids'],'exact declared IDs',ident)
        check(B.E._digest(task['proposals'])==B.E._digest(old['proposals']),'identical entire physical proposal bank',ident)
        check(old['spec']['domains']==task['spec']['domains'],'old O T numeric domains exact',ident)
        check(task['u_regions'][:len(old['u_regions'])]==old['u_regions'],'old U regions exact',ident)
        filtered=V.G.filter_grazing_BI_reuse(old['spec']['visible'],old['boundary_merged'],task['states'],
            gate['new_U_native_ids'],initial_self_hidden=task['initial_self_hidden'])
        check(filtered['visible']==task['spec']['visible'],'retained q covariance weight exact',ident)
        check(filtered['boundary_observations']==task['boundary_merged'],'source and mate U lines filtered exact',ident)
        check(B.E._digest(row['arms']['SAME_FROZEN_E'])==B.E._digest(old['baseline']),'old e baseline exact',ident)
        for name in V.ARMS[1:]:
            arm=row['arms'][name];family=task['families'][arm['WD_family']];spec=task['spec'];pose=arm['actual_pose']
            bundle=B.AUTH.authenticate_supported_native_edges(family['native_xyz'],task['states'],spec['visible'],task['boundary_merged'],
                spec['image_hw'],initial_self_hidden=task['initial_self_hidden'],image_binding=dict(sha256=task['image_sha256'],image_hw=spec['image_hw']),config=spec['config'],**angle)
            problem=B.NATIVE.prepare_native_edge_problem(family['native_xyz'],spec['K'],task['states'],spec['visible'],spec['domains'],
                spec['image_hw'],family['coarse_pose'],config=spec['config'],boundary_observations=task['boundary_merged'],u_regions=task['u_regions'],edge_bundle=bundle)
            fresh=problem.evaluate(pose['R_cf'],pose['centroid'])
            if arm['accepted']:
                check(fresh['accepted'],'current hard guards recomputed',ident,name)
                difference=abs(fresh['objective']-arm['solution']['objective']);maximum=max(maximum,difference)
                check(difference<1e-8,'same current likelihood',ident,name)
            else:
                own=task['families']['ORIGINAL_WD']['coarse_pose']
                check(arm['WD_family']=='ORIGINAL_WD'and pose['R_cf']==own['R']and pose['centroid']==own['t'],'own coarse fallback',ident,name)
            for k,state in enumerate(task['states']):
                if state=='U':check(arm['keypoint_readout']['reported_points'][k]is None,'all U region only',ident,name)
                elif state=='S':check(arm['keypoint_readout']['point_sources'][k]=='POSE_ONLY_MODEL_PROJECTION','all S model only',ident,name)
    check(total==31,'all31 grazing changes retained')
    for key in('source_sha256','input_sha256','protected_user_sha256'):B.E._verify_sources(protocol[key])
    dest.mkdir(parents=True)
    C.write(dest/'AUDIT.json',dict(schema='INITIAL_GRAZING_CACHED_CONTROL_ROOT_AUDIT_V18',checks=checks,failures=failures,
        failure_count=len(failures),rows=238,new_U_count=total,maximum_recomputed_objective_difference=maximum,
        no_reference_labels_images_NN_or_fits=True,source=B.E._binding(Path(__file__)),prediction_seal=B.E._binding(output/'PREDICTION_SEAL.json')))
    C.write(dest/'SEAL.json',dict(audit_sha256=C.sha(dest/'AUDIT.json')))
    print(dict(checks=checks,failures=len(failures),new_U=total,objective_difference=maximum),flush=True)
    if failures:raise ValueError('Independent grazing audit failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=V.OUTPUT)
    a=p.parse_args();main(a.output)
