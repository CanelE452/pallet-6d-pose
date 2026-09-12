"""Recompute paired raw-error endpoints and frozen C success rule."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.contracts import ROOT,RAW,DOC,write,sha
sys.path.insert(0,str(ROOT))
from scripts.paper.framing_closure_v1.static_missing_stat_audit import ranking

BASE=RAW/'C_geometry_preserving_da'
OUT=DOC/'C_geometry_preserving_da'

def read(path):return json.loads(path.read_text())
def errors(row):return np.array([float(v) for v in row['top_keypoint_supervised_errors_px'].split(';') if v])

def geometry(rows,keys):
    pools=[errors(rows[k]) for k in keys if len(errors(rows[k]))]
    pooled=np.concatenate(pools) if pools else np.array([])
    means=[]
    for k in keys:
        e=errors(rows[k])
        if len(e):
            with Image.open(ROOT/rows[k]['image']) as image:diagonal=np.hypot(*image.size)
            means.append(float(e.mean()/diagonal))
    return dict(frames=len(keys),supervised_corners=len(pooled),median_px=float(np.median(pooled)),
        p90_px=float(np.quantile(pooled,.9)),gross20=float((pooled>20).mean()),frame_mean_raw_diagonal=float(np.mean(means)),
        proj5=float((pooled<=5).mean()),proj10=float((pooled<=10).mean()),proj20=float((pooled<=20).mean()))

def main():
    audits={};per=[];all_rows={};pose={};exposure={}
    for seed in [1,2,3]:
        for arm in ['C0','C1','C2']:
            name=f'{arm}_seed{seed}';d=BASE/name;e=d/'evaluation_pose'
            audit=read(d/'TRAINING_AUDIT.json');assert audit['optimizer_updates']==900
            audits[name]=audit;exposure[name]=read(d/'EXPOSURE.json')
            report=read(e/'PAPER_2D.json');m=report['metrics']['box_and_keypoint_2d']
            rows={r['frame_id']:r for r in csv.DictReader((e/'PAPER_2D_per_frame.csv').open())}
            all_rows[name]=rows;positive=[r for r in rows.values() if r['kind']=='POSITIVE'];negative=[r for r in rows.values() if r['kind']=='NEGATIVE']
            scores=lambda rr:np.array([float(r['top_score'] or 0.) for r in rr])
            p=read(e/f'POSE_EVALUATION_{name}.json')['paths']['MAIN'];pose[name]=p
            matched=[r['frame_id'] for r in positive if r['top_iou50_match']=='True' and len(errors(r))]
            gm=geometry(rows,matched)
            # CSV errors are rounded to 6 decimals by the unchanged evaluator.
            assert abs(gm['median_px']-m['keypoint_location_median_px'])<1e-5
            assert abs(gm['p90_px']-m['keypoint_location_p90_px'])<1e-5
            night=[r for r in positive if r['domain']=='NIGHT']
            per.append(dict(arm=arm,seed=seed,ap50=m['box_ap50'],ap50_95=m['box_ap50_95'],
                Det=float(np.mean([r['top_iou50_match']=='True' for r in positive])),
                night_N=len(night),night_Det=float(np.mean([r['top_iou50_match']=='True' for r in night])),
                **ranking(scores(positive),scores(negative)),geometry=gm,pose=p['ALL'],pose_coverage=p['coverage']))
    assert len({v['init_state_sha256'] for v in audits.values()})==1
    for seed in [1,2,3]:
        for step in range(900):
            assert len({exposure[f'{a}_seed{seed}'][step]['synthetic'] for a in ['C0','C1','C2']})==1
            assert exposure[f'C1_seed{seed}'][step]['real']==exposure[f'C2_seed{seed}'][step]['real']
    paired=[]
    for seed in [1,2,3]:
        names=[f'{a}_seed{seed}' for a in ['C0','C1','C2']]
        common=set.intersection(*[{k for k,r in all_rows[n].items() if r['kind']=='POSITIVE' and r['top_iou50_match']=='True' and len(errors(r))} for n in names])
        paired.append(dict(seed=seed,common_frames=len(common),arms={a:geometry(all_rows[f'{a}_seed{seed}'],sorted(common)) for a in ['C0','C1','C2']}))
    by={a:[r for r in per if r['arm']==a] for a in ['C0','C1','C2']}
    means={a:{k:float(np.mean([r[k] for r in rr])) for k in ['ap50','ap50_95','Det','auroc','fpr95','pose_coverage']} for a,rr in by.items()}
    for a in means:
        means[a]['paired_geometry']={k:float(np.mean([r['arms'][a][k] for r in paired])) for k in ['median_px','p90_px','gross20','frame_mean_raw_diagonal']}
        means[a]['pose']={k:float(np.mean([r['pose'][k] for r in by[a]])) for k in ['rotation_median_deg','yaw_median_deg','translation_median_cm','iou3d_median','add_sym_auc']}
    c0,c1,c2=[means[a] for a in ['C0','C1','C2']]
    conventional_gain=c1['ap50_95']-c0['ap50_95'];detection_gain=c2['ap50_95']-c0['ap50_95']
    gates=dict(detection_improvement=detection_gain>0,recovery_70pct=conventional_gain<=0 or detection_gain>=.7*conventional_gain,
        **{k+'_nonworse':c2['paired_geometry'][k]<=c0['paired_geometry'][k] for k in ['median_px','p90_px','gross20']},
        pose_safe=all(c2['pose'][k]<=1.1*c0['pose'][k] for k in ['translation_median_cm','yaw_median_deg']) and c2['pose_coverage']>=c0['pose_coverage']-.01)
    passed=all(gates.values())
    write(OUT/'PER_SEED.json',dict(per_seed=per,paired=paired,seed_means=means,
        metric_notes=dict(Det='top-score IoU>=0.5 matched rate; fixed inference conf0.001',ranking='unchanged historical paper ranking function, including historical stable tie ordering',geometry='supervised0..8 pooled raw errors on 3-arm common frames',pose='unchanged MAIN, not oracle')))
    write(OUT/'TRAINING_AUDIT.json',dict(status='PASS',audits=audits,total_optimizer_updates=8100,
        init_parity=True,actual_augmented_batch_hash_parity=True,real_labels='frozen273 pseudo only',last_only=True))
    write(OUT/'VERDICT.json',dict(status='PASS' if passed else 'FAIL',verdict='C_GEOMETRY_PRESERVING_DA_SIGNAL' if passed else 'C_GEOMETRY_PRESERVING_DA_FAIL',
        gates=gates,C1_minus_C0_detection=conventional_gain,C2_minus_C0_detection=detection_gain,
        recovery_fraction=detection_gain/conventional_gain if conventional_gain>0 else None,evidence_level='DEVELOPMENT' if passed else 'NEGATIVE_DEVELOPMENT',
        adapter_implemented=False,independent_confirmation_status='NOT_RUN'))
    print(dict(gates=gates,passed=passed,means=means),flush=True)

if __name__=='__main__':main()
