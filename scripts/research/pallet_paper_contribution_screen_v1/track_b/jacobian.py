"""Independent 1px check of a 0.01px local numerical alignment Jacobian.

Every call includes the frozen prediction-only W/D selector; its discontinuities
are not hidden by supplying the GT axis. Yaw is compared modulo pi (C2).
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.contracts import ROOT,RAW,DOC,sha,write
sys.path.insert(0,str(ROOT))
from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
from scripts.annotate.canonicalize_fourfold_yaw import yaw_of

OUT=DOC/'B_alignment_loss'

def delta(a,b):
    v=a-b;v[2]=(v[2]+np.pi/2)%np.pi-np.pi/2
    return v

def state(points,K,dims):
    center=points.mean(0)
    result=select_pnp_hypotheses(np.vstack([points,center]),K,dict(x=float(dims[0]),y=float(dims[1]),z=float(dims[2])),None)
    matches=[h for h in result.hypotheses if h.name==result.selected_hypothesis and h.success]
    if not matches or not matches[0].canonical_candidates:raise ValueError('No selected canonical candidate')
    h=matches[0];p=h.canonical_candidates[0]
    return np.array([p.translation[0],p.translation[2],yaw_of(p.rotation)]),h.name

def main():
    tablepath=ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
    population=ROOT/'data/pallet/results/pallet_symmetry_dht_local_v1/export/calibration.json'
    write(OUT/'PROTOCOL_LOCK.json',dict(status='LOCKED_BEFORE_MEASUREMENT',population_sha256=sha(population),
        geometry_table_sha256=sha(tablepath),frames=256,center8_excluded=True,derivative_delta_px=.01,verification_delta_px=1.,
        adapter='frozen select_pnp_hypotheses, first signed canonical candidate, x/z translation and deployment yaw_of; C2 angle differences modulo pi',
        finite_min=.995,relative_median_max=.1,relative_p90_max=.25,catastrophic_sample_max=.01,
        relative_floor=[1e-6,1e-6,1e-6],catastrophic='any coordinate/state relative residual>1 AND actual change>1mm or0.1deg',
        aggregate='each of x,z,yaw must pass both median and P90; failed evaluations retained in finite denominator',
        loss_scales=dict(x_m=.01,z_m=.01,yaw_rad=float(np.deg2rad(1))),loss_weights=[1,1,1],lambda_gradient_ratio=.25,
        B1='existing LC-derived baseline, not novel',no_real_images_or_labels=True))
    table=np.load(tablepath);indices={str(s):i for i,s in enumerate(table['stems'])}
    records=json.loads(population.read_text())['records'];assert len(records)==256
    rows=[];errors=[];yaw_abs=[];finite=0;attempted=256*32;bad=0
    for rec in records:
        i=indices[rec['frame_id']];fx,fy,cx,cy=table['K'][i]
        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
        camera=table['Xcf'][i]@table['R'][i].T+table['t'][i]
        u=camera[:,:2]/camera[:,2:];u=u*np.array([fx,fy])+np.array([cx,cy])
        dims=table['dims'][i]
        row=dict(id=rec['frame_id'],valid=False,catastrophic=True)
        try:
            base,name=state(u,K,dims);J=np.zeros((3,16))
            for j in range(16):
                plus=u.copy();minus=u.copy();plus.flat[j]+=.01;minus.flat[j]-=.01
                J[:,j]=delta(state(plus,K,dims)[0],state(minus,K,dims)[0])/.02
            condition=float(np.linalg.cond(J));local=[];switches=0;cat=False
            for j in range(16):
                for sign in [-1,1]:
                    pert=u.copy();pert.flat[j]+=sign
                    try:
                        actual,branch=state(pert,K,dims);actual=delta(actual,base);estimate=J[:,j]*sign
                        residual=np.abs(estimate-actual);rel=residual/np.maximum(np.abs(actual),1e-6)
                        if not np.isfinite(rel).all():continue
                        finite+=1;errors.append(rel);local.append(rel);yaw_abs.append(float(residual[2]))
                        switches+=branch!=name
                        cat|=bool(((rel>1)&(np.abs(actual)>np.array([.001,.001,np.deg2rad(.1)]))).any())
                    except (ValueError,RuntimeError):cat=True
            row.update(valid=True,condition_number=condition if np.isfinite(condition) else None,
                ill_conditioned=bool(not np.isfinite(condition) or condition>1e6),finite_perturbations=len(local),branch_switches=switches,
                catastrophic=cat,relative_median=np.median(local,axis=0).tolist() if local else None)
        except (ValueError,RuntimeError) as exc:row['error']=str(exc)
        bad+=row['catastrophic'];rows.append(row)
        if len(rows)%32==0:print('B Jacobian',len(rows),'/256',flush=True)
    errors=np.asarray(errors);med=np.median(errors,axis=0);p90=np.quantile(errors,.9,axis=0)
    passed=finite/attempted>=.995 and bool((med<=.1).all()) and bool((p90<=.25).all()) and bad/256<=.01
    write(RAW/'B_alignment_loss/PER_FRAME.json',rows)
    write(OUT/'MECHANISM_RESULT.json',dict(status='PASS' if passed else 'FAIL',
        verdict='B_JACOBIAN_PASS_REQUIRES_GRADIENT_GATE' if passed else 'B_JACOBIAN_INVALID',
        finite_rate=finite/attempted,relative_median_xyz_yaw=med.tolist(),relative_p90_xyz_yaw=p90.tolist(),
        yaw_absolute_error_median_rad=float(np.median(yaw_abs)),yaw_absolute_error_p90_rad=float(np.quantile(yaw_abs,.9)),
        catastrophic_sample_fraction=bad/256,ill_conditioned_fraction=sum(r.get('ill_conditioned',True) for r in rows)/256,
        student_optimizer_updates=0,gradient_gate_status='NOT_RUN',evidence_level='MECHANISM_ONLY'))

if __name__=='__main__':main()
