"""Rebuild selected v18 likelihoods and guards after complete238 sealing."""
from pathlib import Path
from collections import Counter
import argparse
import math
import numpy as np

from . import evaluate_image_chain_mixtures_v4_18 as V


def main(output=V.OUTPUT):
    output=Path(output);dest=output/'independent_root_audit_v1'
    if dest.exists():raise FileExistsError('Preserve prior audit')
    C,E=V.C,V.E;protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREDICTION_SEAL.json')
    if seal['rows']!=238 or C.sha(output/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']:
        raise ValueError('Complete238 prediction seal required')
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    inputs={r['id']:r for r in C.rows(output/'INPUTS.jsonl.gz')};rows=list(C.rows(output/'PREDICTIONS.jsonl.gz'))
    params=V.CAL.mixture_noise_parameters(V._verified_model());checks=0;failures=[];maximum=0.
    def check(ok,what,ident=None,arm=None):
        nonlocal checks
        checks+=1
        if not bool(ok):failures.append(dict(check=what,id=ident,arm=arm))
    check([r['id']for r in rows]==protocol['ids'],'ordered complete238')
    check(Counter(r['partition']for r in rows)==Counter(CAL=46,TEST=49,REMAINDER=143),'46+49+143 severe0')
    for row in rows:
        task=inputs[row['id']];ident=row['id'];check(E._digest(row['arms']['SAME_FROZEN_E'])==E._digest(task['baseline']),'e exact',ident)
        for name in V.ARMS[1:]:
            arm=row['arms'][name];pose=arm['actual_pose'];family=task['families'][arm['WD_family']];spec=task['spec']
            base=V.BASE.NATIVE.prepare_native_edge_problem(family['native_xyz'],spec['K'],task['states'],spec['visible'],
                spec['domains'],spec['image_hw'],family['coarse_pose'],config=spec['config'],
                boundary_observations=task['boundary_merged'],u_regions=task['u_regions'],edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))
            bundle=V.MIX.authenticate_image_chain_groups(base,task['new_chain_observation'],
                initial_self_hidden=task['initial_self_hidden'],image_binding=dict(sha256=task['image_sha256'],image_hw=spec['image_hw']),**params)
            problem=V.MIX.ImageChainMixtureProblem(base,bundle);fresh=problem.evaluate(pose['R_cf'],pose['centroid'])
            check(np.allclose(np.asarray(pose['R_cf'])@family['physical_registration_Q'],pose['R_physical'],atol=1e-10,rtol=0),'physical Q',ident,name)
            if arm['accepted']:
                check(fresh['accepted'],'current guards recomputed',ident,name)
                difference=abs(fresh['objective']-arm['solution']['objective']);maximum=max(maximum,difference)
                check(difference<1e-8,'fresh new mixture objective',ident,name)
                check(fresh['replaced_old_normal_factor_count']==len(bundle['replaces_normal_factor_refs']),'normal refs replaced once',ident,name)
                check(fresh['full_BI_likelihood_revocations']==0,'no silent completeBI revocation',ident,name)
            else:
                coarse=task['families']['ORIGINAL_WD']['coarse_pose']
                check(arm['WD_family']=='ORIGINAL_WD'and pose['R_cf']==coarse['R']and pose['centroid']==coarse['t'],'own coarse fallback only',ident,name)
            readout=arm['keypoint_readout']
            for k,state in enumerate(task['states']):
                if state=='S':check(readout['point_sources'][k]=='POSE_ONLY_MODEL_PROJECTION','S pose only',ident,name)
                elif state=='U':check(readout['reported_points'][k]is None,'U only region',ident,name)
            selection=arm['candidate_selection'];candidates=selection['candidates']
            chosen=next((c for c in candidates if c['name']==selection['selected_name']),None)
            if chosen is not None:
                if name==V.ARMS[1]:
                    minimum=min(c['native_energy']for c in candidates)
                    check(chosen['native_energy']<=minimum+1e-10,'native selection minimum',ident,name)
                    for c in candidates:check(abs(math.fsum(c['native_cost_vector'])-c['native_energy'])<1e-10,'native sum',ident,name)
                else:
                    check(chosen['total_energy']<=min(c['total_energy']for c in candidates)+1e-10,'native+mask minimum',ident,name)
                    for c in candidates:check(abs(c['native_energy']+selection['mask_weight']*c['global_mask_cost']-c['total_energy'])<1e-10,'explicit frozen mask weight',ident,name)
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    dest.mkdir(parents=True)
    C.write(dest/'AUDIT.json',dict(schema='NEW_NMS_MIXTURE_CONTROL_INDEPENDENT_ROOT_AUDIT_V18',checks=checks,
        failures=failures,failure_count=len(failures),maximum_recomputed_objective_difference=maximum,
        rows=238,CAL_not_primary=True,primary192=192,severe=0,new_fits_or_RGB_or_NN_calls=0,
        reference_labels_decoded=0,source=E._binding(Path(__file__)),prediction_seal=E._binding(output/'PREDICTION_SEAL.json')))
    C.write(dest/'SEAL.json',dict(audit_sha256=C.sha(dest/'AUDIT.json')))
    print(dict(checks=checks,failures=len(failures),objective_difference=maximum),flush=True)
    if failures:raise ValueError('New NMS mixture audit failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=V.OUTPUT)
    a=p.parse_args();main(a.output)
