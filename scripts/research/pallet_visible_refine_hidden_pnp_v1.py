"""R0 pose -> frozen self-visibility -> visible refinement -> hidden-only PnP fill."""
import copy
from collections import Counter
import html
import json
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
from scripts.research import pallet_replay_clean19_v1 as P
from scripts.research.pallet_dim_conditioned_p_v1 import pose as Pose
from scripts.research.pallet_cad_refiner_comparison_v1.self_occlusion import visibility
from scripts.paper.pose_metric_closure_v1.run_pose_evaluation import cuboid, solve
from inference import registry_input

ROOT=P.ROOT;NAME='pallet_visible_refine_hidden_pnp_v1'
DOC=ROOT/'_docs/experiments'/NAME;RAW=ROOT/'data/pallet/results'/NAME;OUT=ROOT/'outputs'/NAME
ARMS=('R0','N2_DIM_ONLY','N3_DIM_SYM','TYPE_REPLAY')
read=P.read;bind=P.bind


def save(path,value):
    assert any(path.resolve().is_relative_to(p) for p in (DOC,RAW,OUT))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def pipeline(r0,refined,initial,K,hw):
    """No GT inputs. Never use hidden refiner outputs in visible-only PnP."""
    out=copy.deepcopy(r0);info=dict(applied=False,reason='initial_pose_unavailable',hidden=[],visible=[],used=[],ambiguous=[])
    raw=P.C.selected(r0);corr=P.C.selected(refined)
    if raw is None or corr is None or not initial.get('available'):return out,copy.deepcopy(out),info
    q0=np.array(raw['keypoints_xy'],float);qc=np.array(corr['keypoints_xy'],float)
    x=cuboid(*initial['cf_extents']);R=np.array(initial['R_cf']);t=np.array(initial['centroid'])
    if not np.isfinite(R).all() or not np.isfinite(t).all() or np.min(((R@x.T).T+t)[:,2])<=0:
        info['reason']='initial_pose_nonphysical';return out,copy.deepcopy(out),info
    hidden,visible,cosine=visibility(x,R,t)
    valid=np.isfinite(q0[:8]).all(1)&~(q0[:8]==-1).all(1)
    corrected_valid=np.isfinite(qc[:8]).all(1)&~(qc[:8]==-1).all(1)
    applied_visible=visible&valid&corrected_valid
    q=q0.copy();q[np.flatnonzero(applied_visible)]=qc[:8][applied_visible]
    P.C.selected(out)['keypoints_xy']=q.tolist();visible_only=copy.deepcopy(out)
    info.update(hidden=np.flatnonzero(hidden).tolist(),visible=np.flatnonzero(visible).tolist(),
        ambiguous=np.flatnonzero(~(hidden|visible)).tolist(),corrected=np.flatnonzero(applied_visible).tolist(),
        visibility_cosine=cosine.tolist(),initial_selected_hypothesis=initial['selected_hypothesis'])
    if not hidden.any():info['reason']='no_confident_hidden_corner';return out,visible_only,info
    h,w=hw;inframe=(q[:8,0]>=0)&(q[:8,0]<w)&(q[:8,1]>=0)&(q[:8,1]<h)
    confidence=np.asarray(raw.get('keypoints_conf',[1.]*9)[:8])
    usable=applied_visible&inframe&(confidence>=.5)
    info['used']=np.flatnonzero(usable).tolist()
    assert not (usable&hidden).any()
    if usable.sum()<6:info['reason']='fewer_than_6_reliable_visible_points';return out,visible_only,info
    try:result=solve(x,q[:8],K,usable)
    except (cv2.error,ValueError):result=None
    if result is None:info['reason']='visible_only_PnP_failed';return out,visible_only,info
    R2,t2,residual=result
    if not np.isfinite(R2).all() or not np.isfinite(t2).all() or np.min(((R2@x.T).T+t2)[:,2])<=0:
        info['reason']='refit_pose_nonphysical';return out,visible_only,info
    hidden2,_,_=visibility(x,R2,t2)
    if not np.array_equal(hidden,hidden2):info['reason']='hidden_set_changed';return out,visible_only,info
    projected=cv2.projectPoints(x,cv2.Rodrigues(R2)[0],t2,K,None)[0].reshape(-1,2)
    if not np.isfinite(projected).all():info['reason']='projection_nonfinite';return out,visible_only,info
    # Preserve invalid R0 keypoints rather than fabricate detections.
    replacement=hidden&valid;q[np.flatnonzero(replacement)]=projected[replacement]
    P.C.selected(out)['keypoints_xy']=q.tolist()
    np.testing.assert_array_equal(q[np.r_[~replacement,True]],np.array(P.C.selected(visible_only)['keypoints_xy'])[np.r_[~replacement,True]])
    info.update(applied=True,reason='hidden_only_reprojected',replaced=np.flatnonzero(replacement).tolist(),
        refit_mean_residual_px=float(residual),refit_R=R2.tolist(),refit_t=t2.tolist())
    return out,visible_only,info


def tests():
    K=np.array([[600.,0,320.],[0,600.,240.],[0,0,1.]])
    x=cuboid(1.1,.11,1.3);rv=np.array([.5,.4,0.]);R=cv2.Rodrigues(rv)[0];t=np.array([0.,0.,3.])
    pts=cv2.projectPoints(x,rv,t,K,None)[0].reshape(-1,2);h,v,_=visibility(x,R,t);assert h.sum()==1
    q=np.vstack([pts,[[320.,240.]]]);q[:8][h]+=[30.,-20.];q[:8][v]+=[4.,-3.]
    base=dict(selected_index=0,candidates=[dict(keypoints_xy=q.tolist(),keypoints_conf=[1.]*9)])
    refined=copy.deepcopy(base);r=np.vstack([pts,[[999.,999.]]]);r[:8][h]=[9999.,9999.];P.C.selected(refined)['keypoints_xy']=r.tolist()
    initial=dict(available=True,cf_extents=[1.1,.11,1.3],R_cf=R.tolist(),centroid=t.tolist(),selected_hypothesis='test')
    after,masked,info=pipeline(base,refined,initial,K,(480,640));assert info['applied'],info
    np.testing.assert_allclose(np.array(P.C.selected(after)['keypoints_xy'])[:8],pts,atol=1e-5)
    np.testing.assert_array_equal(np.array(P.C.selected(masked)['keypoints_xy'])[:8][h],q[:8][h])
    assert P.C.selected(after)['keypoints_xy'][8]==[320.,240.]
    refined2=copy.deepcopy(refined);P.C.selected(refined2)['keypoints_xy'][np.flatnonzero(h)[0]]=[-8888.,5555.]
    assert pipeline(base,refined2,initial,K,(480,640))[0]==after
    fail=copy.deepcopy(initial);fail['available']=False;assert pipeline(base,refined,fail,K,(480,640))[0]==base
    low=copy.deepcopy(base);P.C.selected(low)['keypoints_conf']=[0.]*9
    a,b,i=pipeline(low,refined,initial,K,(480,640));assert not i['applied'] and a==b
    return dict(synthetic_hidden_recovery=True,hidden_refiner_output_ignored=True,
        visible_only_refit=True,center_preserved=True,initial_failure_R0_fallback=True,insufficient_points_no_hidden_replacement=True)


def main():
    cv2.setNumThreads(1)
    assert not any(p.exists() for p in (DOC,RAW,OUT))
    unit=tests()
    for p in (DOC,RAW,OUT):p.mkdir(parents=True)
    split=read(P.DOC/'SPLIT.json');records=split['evaluation'];assert len(records)==300
    base=read(P.RAW/'PREDICTIONS.json')['predictions']
    rawpred={a:base[a] for a in ARMS if a!='TYPE_REPLAY'};rawpred['TYPE_REPLAY']={}
    protected=[bind(P.DOC/'SPLIT.json'),bind(P.RAW/'PREDICTIONS.json'),bind(Path(__file__)),bind(Path(Pose.__file__))]
    for material in ('plastic','wood'):
        path=ROOT/f'data/pallet/results/pallet_replay_by_type_v1/{material}/PREDICTIONS.json'
        rawpred['TYPE_REPLAY'].update(read(path)['predictions']['CLEAN19_REPLAY']);protected.append(bind(path))
    assert all(set(p)=={r['id'] for r in records} for p in rawpred.values())
    # Inference metadata: registry dimensions and camera intrinsics ONLY; no target or annotated pose.
    cache=read(P.C.E.DOC/'DEV_CACHE_COMPLETE.json');types={r['id']:r['object_type'] for r in cache['records']}
    inputs=[]
    for r in records:
        cam=read(ROOT/r['annotation']['path'])['camera_data'];intr=cam['intrinsics']
        dims,_=registry_input(types[r['id']])
        K=[[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]]
        inputs.append(dict(id=r['id'],K=K,xyz=dims[[0,2,1]].tolist(),hw=[cam['height'],cam['width']]))
        protected.extend([r['image'],r['annotation']])
    save(RAW/'INFERENCE_METADATA.json',inputs)
    save(DOC/'PROTOCOL.json',dict(steps=['initial R0-only PnP','freeze cuboid self-visibility','apply refiner only to confident non-self-hidden corners','refit using corrected confident visible corners only','replace hidden only by reprojection'],
        no_training=True,no_new_neural_inference=True,frames=300,initial_pose_shared_by_all_arms=True,
        arms=list(ARMS),angle_margin_degrees=2,minimum_refit_points=6,keypoint_confidence=.5,
        dimensions_hypothesis='initial R0 prediction-only W/D selection then frozen',
        ambiguity='within +/-2deg keep R0; do not include in refit',
        fallback='initial pose invalid: R0 unchanged; refit invalid/insufficient/hidden-set-changed: keep visible refinement, hidden R0',
        camera_intrinsics='existing supplied calibration; not independently calibrated in this experiment',
        caveats=['convex cuboid proxy, not actual mesh; external occlusion not identified','existing-reference metric, not independent hidden 3D ground truth','same-session adaptation; no final model auto replacement'],
        tests=unit,sources=protected))
    output={};decisions={};initialposes={}
    for a in ARMS:
        output[a+'_DIRECT']=rawpred[a];output[a+'_VISIBLE_ONLY']={};output[a+'_PIPELINE']={};decisions[a]={}
    for k,r in enumerate(inputs):
        fid=r['id'];c=P.C.selected(base['R0'][fid]);K=np.array(r['K'])
        points=None if c is None else np.array(c['keypoints_xy'],float)
        if points is not None:points[(points==-1).all(1)]=np.nan
        initial=Pose.infer(points,K,np.array(r['xyz']),False);initialposes[fid]=initial
        for a in ARMS:
            final,visible,info=pipeline(base['R0'][fid],rawpred[a][fid],initial,K,r['hw'])
            P.assert_preserved(base['R0'][fid],final)
            output[a+'_PIPELINE'][fid]=final;output[a+'_VISIBLE_ONLY'][fid]=visible;decisions[a][fid]=info
        if (k+1)%50==0:print('PIPELINE',k+1,300,flush=True)
    save(RAW/'INITIAL_R0_POSES.json',initialposes)
    save(RAW/'DECISIONS.json',decisions)
    save(RAW/'PREDICTIONS.json',dict(predictions=output,GT_input=False))
    save(DOC/'PREDICTION_LOCK.json',dict(predictions=bind(RAW/'PREDICTIONS.json'),metadata=bind(RAW/'INFERENCE_METADATA.json'),frozen_before_scoring=True))
    # Only now access scoring targets; initial pose and visibility never use them.
    truth=read(P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json');legacy=read(P.RAW/'FRAME_METRICS.json')['R0']
    metrics={a:{} for a in output}
    for a,pp in output.items():
        for fid,p in pp.items():
            t=truth[fid];c=P.C.selected(p);q=np.full((9,2),np.nan) if c is None else c['keypoints_xy']
            m=P.M.measure(q,t['gt'],t['valid'],t['permutations'],t['hw'],legacy[fid]['matched'],legacy[fid]['detected'])
            metrics[a][fid]=dict(id=fid,**m)
    groups={'ALL300':records,**{m:[r for r in records if r['object_type']==m] for m in ('plastic','wood')},
            **{s:[r for r in records if r['severity']==s] for s in P.SEVERITIES}}
    summary={}
    for g,rows in groups.items():
        summary[g]={}
        for a in output:
            rr=[metrics[a][r['id']] for r in rows];s=P.M.summary(rr)
            s['gross20_count']=sum(e>20 for row in rr for e in row.get('errors',[]));summary[g][a]=s
        assert len({(s['corners'],s['total_frames'],s['matched']) for s in summary[g].values()})==1
    stats={a:dict(reasons=dict(Counter(i['reason'] for i in dd.values())),applied=sum(i['applied'] for i in dd.values()),
                  hidden_counts=dict(Counter(len(i['hidden']) for i in dd.values()))) for a,dd in decisions.items()}
    save(DOC/'RESULTS.json',summary);save(DOC/'APPLICATION_COUNTS.json',stats);save(RAW/'FRAME_METRICS.json',metrics)
    report(summary,stats,records,output,decisions,metrics)
    for b in protected:P.N.F.verify(b)
    save(DOC/'AUDIT.json',dict(complete=True,tests=unit,frames=300,training=0,neural_inference=0,
        initial_visibility_from_R0=True,hidden_not_used_in_refit=True,outputs_locked_before_scoring=True,
        same_evaluation_denominator=True,previous_files_unchanged=True))
    print(json.dumps({g:{a:s['PCK']['10'] for a,s in v.items() if a.startswith('TYPE')} for g,v in summary.items()},indent=2))


def report(summary,stats,records,output,decisions,metrics):
    lines=['# 초기R0 자기 가림 → 보이는 점 보정 → 숨은 점 PnP 대체','',
        '사용자가 지정한 처리 순서로 적용했다. 초기 R0 PnP/가림 마스크는 모든 모델에서 동일하다. 보정기의 숨은 점 출력은 버린다. 비가림 보정점으로만 두 번째 PnP를 풀어 숨은 점만 채운다. 추가 학습0, 기존 예측 재사용.', '',
        '±2도 경계는 미확정이므로 R0 유지. 재추정은 화면내부·confidence≥0.5 비가림6점 이상에서만 수행. 실패 시 비가림 보정은 유지하고 가림점은 R0 유지. 이 fallback은 보정기 전체 출력으로 되돌리는 것이 아니다.', '',
        'DIRECT=기존 전체코너 보정 출력. VISIBLE_ONLY=숨은점/경계점은 R0, 확실한 비가림점만 보정. PIPELINE=VISIBLE_ONLY 후 숨은점 재투영.', '']
    fmt=lambda x:'—' if x is None else f'{x:.2f}'
    for g,ss in summary.items():
        lines+=['## '+g,'','| 방법 | PCK10% | PCK20% | observed med px | P90 px | >20 코너 |','|---|---:|---:|---:|---:|---:|']
        for a,s in ss.items():
            lines.append(f'| {a} | {fmt(100*s["PCK"]["10"] if s["PCK"]["10"] is not None else None)} | {fmt(100*s["PCK"]["20"] if s["PCK"]["20"] is not None else None)} | {fmt(s["matched_pooled_corner8_median_px"])} | {fmt(s["matched_pooled_corner8_P90_px"])} | {s["gross20_count"]} |')
        lines.append('')
    lines+=['## 적용 수','',*['- '+a+': '+str(s) for a,s in stats.items()], '',
        '실제 팔레트의 구멍·외부가림을 추정하는 모델은 아니며 알려진 치수의 직육면체 자기 가림 근사다. K는 기존제공값이다. 결과는 기존 reference와의2D일치도이지 독립 물리 pose 정확도가 아니다. 매칭 실패는 기존 penalty 유지. 같은세션 적응·재사용 평가이며 모델 자동교체 없음.', '',
        '로컬 단계별 이미지: outputs/pallet_visible_refine_hidden_pnp_v1/GALLERY.html']
    save(DOC/'RESULTS_KO.md','\n'.join(lines)+'\n')
    # All 300 frames, no cherry-picking. Render native stage outputs; reference is not overlaid.
    (OUT/'images').mkdir();panels=[]
    for i,r in enumerate(records):
        fid=r['id'];name=f'{i:03d}.png';shutil.copyfile(ROOT/r['image']['path'],OUT/'images'/name)
        info=decisions['TYPE_REPLAY'][fid]
        panels.append(f'<section><h2>{i+1}/300 · {html.escape(fid)} · {r["object_type"]}</h2><p>R0 추정 self-hidden: {info["hidden"]} · 비가림: {info["visible"]} · 재추정 입력: {info["used"]} · {info["reason"]}</p><div class="grid">')
        for arm,label in [('R0_DIRECT','1. R0'),('TYPE_REPLAY_DIRECT','참고: 보정기 전체 출력'),('TYPE_REPLAY_VISIBLE_ONLY','2. 비가림만 보정'),('TYPE_REPLAY_PIPELINE','3. 가림점 PnP 대체')]:
            c=P.C.selected(output[arm][fid]);points=[]
            if c:
                xy=c['keypoints_xy']
                for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
                    if np.isfinite([*xy[a],*xy[b]]).all():points.append(f'<line x1="{xy[a][0]}" y1="{xy[a][1]}" x2="{xy[b][0]}" y2="{xy[b][1]}" stroke="#81cfef" stroke-width="1"/>')
                for j,(x,y) in enumerate(xy[:8]):
                    if not np.isfinite([x,y]).all():continue
                    color='#ff8a4c' if j in info['hidden'] else '#55ff9a'
                    points.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/><text x="{x+5}" y="{y-4}" fill="{color}" stroke="#142631" stroke-width=".3">P{j}</text>')
            m=metrics[arm][fid];err=m.get('errors',[])
            panels.append(f'<article><h3>{label}</h3><p>10px 이내 {sum(e<=10 for e in err)}/{len(err)}</p><svg viewBox="0 0 640 480"><image href="images/{name}" width="640" height="480"/>{"".join(points)}</svg></article>')
        panels.append('</div></section>')
    save(OUT/'GALLERY.html','<!doctype html><meta charset="utf-8"><style>body{background:#142631;color:white;font:16px system-ui}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}svg{width:100%}section{border-top:2px solid #abc;margin-top:30px}</style><h1>R0 → 비가림 보정 → 자기 가림점 PnP 대체 · 전체300장</h1><p>주황=R0 PnP가 추정한 자기 가림점. 초록=나머지. GT좌표 오버레이 없음. 숫자는 예측을 고정한 뒤 채점한 기존 reference 일치도.</p>'+''.join(panels))


if __name__=='__main__':main()
