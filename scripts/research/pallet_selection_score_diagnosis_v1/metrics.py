"""Measure already saved candidates against the unchanged bound reference."""
from pathlib import Path
import argparse
import math
from decimal import Decimal,localcontext

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.research.pallet_statewise_correspondence_diagnosis_v1.analysis import metric
from .common import read,rows,write,write_csv,save_rows,sha,timed_stage
from .runtime_guard import FixedPoseOnly


def independent_error(pose,reference):
    delta=[float(a)-float(b) for a,b in zip(pose['centroid'],reference['t'])]
    translation=100*math.sqrt(math.fsum(x*x for x in delta))
    if reference['R'] is None:return translation,None
    G=np.asarray(reference['R'],float);R=np.asarray(pose['R_physical'],float)
    symmetry=np.diag([-1.,1.,-1.])
    angle=min(float(np.degrees(Rotation.from_matrix((G@S).T@R).magnitude()))
              for S in (np.eye(3),symmetry))
    return translation,angle


def decimal_translation_check(pose,reference,observed):
    """Independent high-precision check with an IEEE arithmetic error bound.

    This is not a pose-acceptance tolerance. Very distant retained candidates
    can have a single float ULP larger than 1e-9cm. Preserve the original strict
    absolute-check outcome, then verify its numerical rounding explanation.
    The bound covers coordinate subtraction, three products/two sums, sqrt and
    multiplication by100. It is derived before comparing the discrepancy.
    """
    with localcontext() as context:
        context.prec=80
        exact=[Decimal.from_float(float(a))-Decimal.from_float(float(b))
               for a,b in zip(pose['centroid'],reference['t'])]
        delta=[float(a)-float(b) for a,b in zip(pose['centroid'],reference['t'])]
        subtraction=sum(abs(Decimal.from_float(d)-e) for d,e in zip(delta,exact))
        norm=sum(Decimal.from_float(d)**2 for d in delta).sqrt()
        truth=sum(d*d for d in exact).sqrt()*100
        u=Decimal(2)**-53;gamma5=5*u/(1-5*u)
        bound=100*(subtraction+gamma5*norm+Decimal.from_float(math.ulp(float(norm))))
        bound+=Decimal.from_float(math.ulp(observed))
        discrepancy=abs(Decimal.from_float(observed)-truth)
        return dict(pass_=discrepancy<=bound,absolute_discrepancy_cm=float(discrepancy),
                    derived_arithmetic_bound_cm=float(bound),truth_cm=str(truth))


def run(parent,selection,output):
    parent,selection,output=map(Path,(parent,selection,output))
    output.mkdir(parents=True,exist_ok=False)
    seal=read(parent/'score_v2/SCORE_SEAL.json')
    reference_path=parent/'score_v2/FIXED_REFERENCES.json'
    assert sha(reference_path)==seal['artifacts']['FIXED_REFERENCES.json']
    references=read(reference_path)['references']
    records=[];json_records=[];issues=[];absolute_discrepancies=[];maxT=maxR=0.
    with timed_stage(output.parent,'fixed_saved_candidate_metrics',new_pose_calls=0):
        with FixedPoseOnly() as guard:
            for candidate in rows(selection/'RETAINED_CANDIDATES.jsonl.gz'):
                ref=references[candidate['partition']][candidate['id']]
                pose={**candidate['pose'],'available':True}
                value=metric(pose,ref)
                T,R=independent_error(pose,ref)
                dt=abs(T-value['translation_cm']);dr=0 if R is None else abs(R-value['rotation_deg'])
                maxT=max(maxT,dt);maxR=max(maxR,dr)
                decimal=decimal_translation_check(pose,ref,value['translation_cm'])
                if dt>1e-9:absolute_discrepancies.append(dict(id=candidate['id'],candidate_id=candidate['candidate_id'],
                    translation_cm=value['translation_cm'],absolute_difference_cm=dt,
                    **{k:v for k,v in decimal.items() if k!='truth_cm'}))
                if not decimal['pass_'] or dr>1e-6 or value['success_le5']!=candidate['reference_success']:
                    issues.append(dict(id=candidate['id'],candidate_id=candidate['candidate_id'],translation_difference=dt,
                        rotation_difference=dr,sealed_success=candidate['reference_success'],recomputed_success=value['success_le5']))
                records.append(dict(id=candidate['id'],index=candidate['index'],partition=candidate['partition'],
                    stage=candidate['stage'],family=candidate['family'],candidate_id=candidate['candidate_id'],
                    translation_cm=value['translation_cm'],rotation_deg=value['rotation_deg'],
                    reference_success=value['success_le5'],pose_available=value['pose_available']))
                json_records.append(dict(id=candidate['id'],candidate_id=candidate['candidate_id'],metric=value))
    assert len(records)==11603
    write_csv(output/'CANDIDATE_METRICS.csv',records)
    save_rows(output/'CANDIDATE_METRICS.jsonl.gz',json_records)
    write(output/'METRIC_CHECK.json',dict(rows=len(records),all_pass=not issues,issues=issues,
        original_reference_success_ids_checked=len(records),independent_formula_rows=len(records),
        max_translation_difference_cm=maxT,max_rotation_difference_deg=maxR,
        initial_absolute_check_cm=1e-9,initial_absolute_check_failed_rows=absolute_discrepancies,
        initial_absolute_check_all_pass=not absolute_discrepancies,
        independent_decimal_arithmetic_check_all_pass=not issues,independent_rotation_tolerance_deg=1e-6,
        original_selector_and_cost_atol_1e9_not_changed=True,
        single_Rt_joint_success=True,physical_C2_only=True,reference_is_geometric_reconstruction=True,
        reference_sha256=sha(reference_path),selection_seal_sha256=sha(selection/'SELECTION_SEAL.json'),
        metric_source_sha256=sha(Path(__file__)),runtime_guard=guard.receipt(),
        new_neural_forward=0,new_PnP_or_optimizer_or_candidate_generation=0,
        artifacts={p.name:sha(p) for p in (output/'CANDIDATE_METRICS.csv',output/'CANDIDATE_METRICS.jsonl.gz')}))
    if issues:raise ValueError('Saved-candidate reference metric binding failed')
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('parent','selection','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();values=run(a.parent,a.selection,a.output)
    print('Saved candidate metrics and independent check PASS:',len(values))
