"""Fixed Hough pseudo-label quality screen; evaluation images never enter training."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts/research/dht_pose_integration_v1'))
from challenge.data_paths import EVAL_CANONICAL, FINAL_TEST
import uncertainty_geometry as U
from fusion import INCIDENT_ROLES
LIVE = ROOT/'data/pallet/results/dht_pose_integration_v1/live'
OLD = LIVE/'uncertainty_fusion_v1'
M4 = ROOT/'data/pallet/results/paper_selftrain_v1/M4_FRAME_RECORDS.json'
ARMS = ['baseline', 'line_uncertainty_only', 'unconditional_lambda1']
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v, (tuple,list)): return [clean(x) for x in v]
    if isinstance(v,np.ndarray): return clean(v.tolist())
    if isinstance(v,np.generic): return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    return v

def write(p,v): Path(p).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2)+'\n')

def setup(out):
    out.mkdir(parents=True,exist_ok=True)
    manifest=read(LIVE/'manifest.json')['records']; f=read(M4)
    fm={r['frame_id']:r for r in f['frames']}
    ids=[r['id'] for r in manifest if r['population']=='real_dev' and r['id'].replace('__',':') in fm]
    sources=[M4,LIVE/'manifest.json',LIVE/'BASELINE_YOLO.json',OLD/'PREDICTIONS.json',OLD/'DHT_UNCERTAINTY.json',OLD/'CALIBRATION.json',OLD/'SELECTION.json',Path(__file__),Path(U.__file__)]
    protocol=dict(experiment='hough_pseudolabel_quality_v1',ids=ids,seeds=[1,2,3],primary_population='original_F4_accepted',
      primary_arm='line_uncertainty_only',arms=ARMS,lambda_fixed=0.0625,max_move=None,
      primary_gt='M4 original gt_xy and gt_supervised, 9 keypoints including unchanged center',
      filter='Original F4 membership fixed BEFORE correction; no threshold change or recomputed flip/removal',
      criteria=dict(mean_improvement_each_seed_min_percent=1,paired_bootstrap_ci_upper_below_zero=True,p90_no_worse_each_seed=True,gross20_frame_count_no_increase_each_seed=True,easy10_point_mean_worsening_max_percent=1),
      bootstrap=dict(repeats=5000,seed=45001,unit='frame resampling within each of 3 fixed sessions; shared 3-seed frame indices; conditional DEV interval, not scene generalization'),
      sensitivity_thresholds_px=[10,20,40],coordinate_preservation='same membership, box, confidence, keypoint validity, center and missing points',
      scope='Existing DEV overlap40 only, expected accepted35. No new test images, no real or synthetic fitting, no student training, no pseudo export to training pool. Does NOT reopen frozen architecture research family. Previous weak-line recipe chosen using synthetic128; it was already observed on DEV52 and is not a prospectively independent method selection.',
      gt_uncertainty='Manual and reconstructed supervision retained, not certified exact. No GT edits. 10/20/40px thresholds are descriptive sensitivity, not tuned accept rules.',
      source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
    if (out/'PROTOCOL.json').exists(): assert read(out/'PROTOCOL.json')==protocol,'Frozen protocol changed'
    else: write(out/'PROTOCOL.json',protocol)
    (out/'PURPOSE.md').write_text('# Hough 의사 라벨 품질 검사\n\n기존 F4 통과 라벨의 좌표를 고정 Hough 규칙으로 보정했을 때 정답 오차와 큰 오류가 줄어드는지 검사한다. 결과는 재사용 DEV 품질 진단이다. 실제 adaptation pool 또는 학생 학습 성능과 구분한다. 평가 이미지는 학습에 넣지 않는다.\n')
    return protocol


def independent_solve(p,valid,lines,vp,vl,lam):
    result=p.copy()
    for k,roles in enumerate(INCIDENT_ROLES):
        if not valid[k]: continue
        # Independently solve absolute-coordinate weighted least squares.
        A=[np.array([1.,0.])/np.sqrt(max(vp[k,0],1)),np.array([0.,1.])/np.sqrt(max(vp[k,1],1))]
        b=[p[k,0]/np.sqrt(max(vp[k,0],1)),p[k,1]/np.sqrt(max(vp[k,1],1))]
        for j,role in enumerate(roles):
            a,z=lines[role]; d=z-a
            if not np.isfinite(d).all() or np.linalg.norm(d)<1e-9: continue
            n=np.array([-d[1],d[0]])/np.linalg.norm(d); scale=np.sqrt(lam/max(vl[k,j],1))
            A.append(n*scale); b.append(np.dot(n,a)*scale)
        result[k]=np.linalg.lstsq(np.asarray(A),np.asarray(b),rcond=None)[0]
    return result


def summarize(err,mask):
    # mask is immutable GT supervision AND observation, same across all arms.
    x=err[mask]; frames=np.flatnonzero(mask.any(1))
    return dict(n_frames=len(err),n_scored_frames=len(frames),n_points=len(x),mean_px=np.mean(x),median_px=np.median(x),p90_px=np.quantile(x,.9),
      bad_frames={str(t):int(np.sum([np.any(err[i,mask[i]]>t) for i in frames])) for t in [10,20,40]},
      gross_points={str(t):int(np.sum(x>t)) for t in [10,20,40]})


def analyze(out,protocol):
    manifest={r['id']:r for r in read(LIVE/'manifest.json')['records']}; f=read(M4); fm={r['frame_id']:r for r in f['frames']}
    preds={r['id']:r for r in read(OLD/'PREDICTIONS.json')['records']}; lines={r['id']:r for r in read(OLD/'DHT_UNCERTAINTY.json')['records']}
    assert f['teacher_sha256']==read(LIVE/'BASELINE_YOLO.json')['weights_sha256']
    assert read(OLD/'SELECTION.json')['families']['line_uncertainty_only']['selected_rule']==dict(lambda_=0) if False else True
    rule=read(OLD/'SELECTION.json')['families']['line_uncertainty_only']['selected_rule']
    assert rule=={'lambda':.0625,'max_move_diagonal_fraction':None}
    rows=[]; max_replay=0.; max_lstsq=0.; max_gt_delta=0.; hashes={}; omitted=[]
    for rid in protocol['ids']:
        r=manifest[rid]; z=fm[rid.replace('__',':')]; p=preds[rid]
        assert r['group'] in EVAL_CANONICAL and r['group'] not in FINAL_TEST
        canonical=ROOT/EVAL_CANONICAL[r['group']]/(Path(r['annotation']).stem+'.json')
        assert canonical.resolve()==Path(r['annotation']).resolve()
        annotation=read(canonical); assert annotation['objects'][0]['split']=='eval'
        for path,digest in [(r['image'],r['image_sha256']),(r['annotation'],r['annotation_sha256'])]:
            assert sha(path)==digest; hashes[str(Path(path).relative_to(ROOT))]=digest
        assert sha(ROOT/z['image_path'])==r['image_sha256']
        base=np.asarray(z['keypoints_xy'],float); valid=np.asarray(z['keypoint_valid'],bool)
        assert np.array_equal(base,np.asarray(p['baseline']['kps'],float),equal_nan=True)
        assert np.array_equal(valid,p['baseline']['kp_valid'])
        gt=np.asarray(z['gt_xy'],float); sup=np.asarray(z['gt_supervised'],bool)
        delta=np.nanmax(np.abs(gt[:8]-np.asarray(r['gt_points'],float))); max_gt_delta=max(max_gt_delta,delta); assert delta==0
        mask=valid&sup&np.isfinite(base).all(1)&np.isfinite(gt).all(1)
        e=np.hypot(*(base-gt).T)
        np.testing.assert_allclose(e[mask],z['errors_annotated_px'],atol=1e-10,rtol=0)
        filter_replay=z['detected'] and z['box_conf']>=.85 and z['valid_corners']>=6 and z['s_remove']<=.05 and z['s_flip']<=.05
        assert bool(filter_replay)==z['verdict']['F4_PROPOSED']
        outputs={}; errors={}
        for seed in protocol['seeds']:
            slot={}; es={}; l=np.asarray(lines[rid]['seeds'][str(seed)]['lines'],float)
            for arm in ARMS:
                pred=base.copy()
                if arm!='baseline':
                    old=p['seeds'][str(seed)][arm]
                    assert np.array_equal(valid,old['kp_valid'])
                    if arm=='line_uncertainty_only':
                        vp=np.asarray(old['point_variance_px2']); vl=np.asarray(old['incident_line_variance_px2']); lam=.0625
                    else: vp=np.ones((9,2)); vl=np.ones((8,2)); lam=1.
                    pred,valid2=U.fuse(base,valid,l,vp,vl,lam,r['width'],r['height'])
                    assert np.array_equal(valid,valid2)
                    max_replay=max(max_replay,float(np.nanmax(np.abs(pred-np.asarray(old['kps'],float)))))
                    check=independent_solve(base,valid,l,vp,vl,lam)
                    max_lstsq=max(max_lstsq,float(np.nanmax(np.abs(check-pred))))
                    np.testing.assert_allclose(pred[8],base[8],equal_nan=True,atol=0,rtol=0)
                    np.testing.assert_allclose(pred[~valid],base[~valid],equal_nan=True,atol=0,rtol=0)
                slot[arm]=pred; es[arm]=np.hypot(*(pred-gt).T)
            outputs[str(seed)]=slot; errors[str(seed)]=es
        rows.append(dict(id=rid,group=r['group'],image=r['image'],gt=gt,gt_supervised=sup,kp_valid=valid,mask=mask,accepted=z['verdict']['F4_PROPOSED'],original_filter_scores={k:z[k] for k in ['box_conf','valid_corners','s_remove','s_flip']},predictions=outputs,errors=errors,lines={s:v['lines'] for s,v in lines[rid]['seeds'].items()}))
    assert len(rows)==40 and sum(r['accepted'] for r in rows)==35
    assert max_replay<1e-9 and max_lstsq<1e-9
    # Serialize row-level evidence, then independently use scalar hypot to audit all errors.
    write(out/'ROWS.json',rows)
    import math
    for r in read(out/'ROWS.json'):
        for seed,arms in r['predictions'].items():
            for arm,pts in arms.items():
                for k,ok in enumerate(r['mask']):
                    if ok:
                        actual=math.hypot(pts[k][0]-r['gt'][k][0],pts[k][1]-r['gt'][k][1])
                        assert abs(actual-r['errors'][seed][arm][k])<1e-10
    results={}
    subsets={'all_overlap':list(range(len(rows))), 'F4_accepted':[i for i,r in enumerate(rows) if r['accepted']], 'F4_rejected':[i for i,r in enumerate(rows) if not r['accepted']]}
    for group in sorted({r['group'] for r in rows}): subsets['accepted/'+group]=[i for i,r in enumerate(rows) if r['accepted'] and r['group']==group]
    for name,ids in subsets.items():
        mask=np.asarray([rows[i]['mask'] for i in ids]); groups=np.asarray([rows[i]['group'] for i in ids]); population={}
        for arm in ARMS:
            per_seed=[]
            for seed in protocol['seeds']:
                a=np.asarray([rows[i]['errors'][str(seed)][arm] for i in ids]); b=np.asarray([rows[i]['errors'][str(seed)]['baseline'] for i in ids]); s=summarize(a,mask)
                easy=mask&(b<=10); s['easy_points']=int(easy.sum()); s['easy_mean_px']=np.mean(a[easy]); s['easy_worsening_percent']=100*(np.mean(a[easy])/np.mean(b[easy])-1)
                s['mean_gain_percent']=100*(1-s['mean_px']/np.mean(b[mask])); s['improved_points_gt_0px']=int(((a<b)&mask).sum()); s['worsened_points_gt_1px']=int(((a>b+1)&mask).sum())
                s['bad_frame_transitions']={}
                for t in [10,20,40]:
                    bad0=np.any(mask&(b>t),axis=1); bad1=np.any(mask&(a>t),axis=1)
                    s['bad_frame_transitions'][str(t)]={'bad_to_good':int((bad0&~bad1).sum()),'good_to_bad':int((~bad0&bad1).sum())}
                s['corner_only']=summarize(a[:,:8],mask[:,:8]); per_seed.append(s)
            diff=np.mean([np.asarray([rows[i]['errors'][str(seed)][arm] for i in ids])-np.asarray([rows[i]['errors'][str(seed)]['baseline'] for i in ids]) for seed in protocol['seeds']],axis=0)
            numerator=np.where(mask,diff,0).sum(1); denom=mask.sum(1); rng=np.random.default_rng(45001); boot=[]
            for _ in range(5000):
                draw=np.concatenate([rng.choice(np.flatnonzero(groups==g),np.sum(groups==g),replace=True) for g in sorted(set(groups))]); boot.append(numerator[draw].sum()/denom[draw].sum())
            population[arm]=dict(per_seed=per_seed,mean_px=np.mean([s['mean_px'] for s in per_seed]),median_px=np.mean([s['median_px'] for s in per_seed]),p90_px=np.mean([s['p90_px'] for s in per_seed]),difference_ci95=np.quantile(boot,[.025,.975]),mean_difference_px=numerator.sum()/denom.sum())
        results[name]=population
    primary=results['F4_accepted']['line_uncertainty_only']; baseline=results['F4_accepted']['baseline']; ss=primary['per_seed']
    gates=dict(mean_gain=all(s['mean_gain_percent']>=1 for s in ss),conditional_ci=primary['difference_ci95'][1]<0,p90=all(s['p90_px']<=b['p90_px'] for s,b in zip(ss,baseline['per_seed'])),gross20=all(s['bad_frames']['20']<=b['bad_frames']['20'] for s,b in zip(ss,baseline['per_seed'])),easy_preservation=all(s['easy_worsening_percent']<=1 for s in ss))
    passed=all(gates.values()); decision='PROMISING_DEV_LABEL_CORRECTION_ONLY' if passed else 'NO_STABLE_PSEUDOLABEL_QUALITY_GAIN_IN_THIS_SCREEN'
    summary=dict(complete=True,n_overlap=40,n_accepted=35,n_excluded_from_old52=12,n_seeds=3,results=results,gates=gates,passed=passed,decision=decision,student_training_updates=0,scope=protocol['scope'])
    write(out/'SUMMARY.json',summary)
    write(out/'AUDIT.json',dict(status='PASS',teacher_coordinates_max_delta_px=0,gt_coordinates_max_delta_px=max_gt_delta,original_error_replay=True,original_F4_membership_replayed=True,weak_and_unconditional_prediction_replay_max_px=max_replay,independent_absolute_lstsq_max_px=max_lstsq,scalar_error_audit=True,center_and_missing_preserved=True,image_annotation_sha256=hashes,source_sha256=protocol['source_sha256']))
    print(json.dumps(clean(dict(decision=decision,gates=gates,primary=primary,baseline=baseline)),ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=ROOT/'data/pallet/results/hough_pseudolabel_quality_v1'); parser.add_argument('--phase',choices=['freeze','evaluate'],default='evaluate'); args=parser.parse_args()
    protocol=setup(args.out)
    if args.phase=='evaluate': analyze(args.out,protocol)
