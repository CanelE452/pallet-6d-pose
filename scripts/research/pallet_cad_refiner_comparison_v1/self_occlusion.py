"""Bounded GT-free cuboid self-occlusion replacement, frozen CAD18 predictions."""
import copy
import html
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_cad_r0_gt_gallery_v1 import build as G
from scripts.self_training_yolo import pseudo_label_filters as F
from scripts.paper.pose_metric_closure_v1.run_pose_evaluation import cuboid,solve

BASE=ROOT/'outputs/pallet_cad_refiner_comparison_v1'
OUT=BASE/'self_occlusion'
ARMS=['R0','N3_DIM_SYM_seed1','REPLAY_RAW']
MARGIN=np.sin(np.deg2rad(2.))


def visibility(x,R,t):
    """A convex-box vertex is hidden iff all three incident faces face away."""
    camera=-R.T@t
    rays=camera[None,:]-x
    cosines=np.sign(x)*rays/np.linalg.norm(rays,axis=1)[:,None]
    best=cosines.max(1)
    return best < -MARGIN,best > MARGIN,best


def correct(candidate,initial,K):
    original=np.array(candidate['keypoints_xy'],float);out=original.copy()
    info=dict(applied=False,reason='initial_pose_unavailable',hidden=[],used=[],before=original.tolist())
    if not initial['available']:return out,info
    x=cuboid(*initial['cf_extents']);R=np.array(initial['R_cf']);t=np.array(initial['centroid'])
    hidden,visible,cosine=visibility(x,R,t)
    info.update(hidden=np.flatnonzero(hidden).tolist(),visibility_cosine=cosine.tolist(),initial_pose=initial)
    if not hidden.any():info['reason']='no_confidently_hidden_corner';return out,info
    valid=np.isfinite(original[:8]).all(1)&~(original[:8]==-1).all(1)
    inframe=(original[:8,0]>=0)&(original[:8,0]<640)&(original[:8,1]>=0)&(original[:8,1]<480)
    usable=visible&valid&inframe&(np.array(candidate['keypoints_conf'][:8])>=.5)
    info['used']=np.flatnonzero(usable).tolist()
    if usable.sum()<6:info['reason']='fewer_than_6_reliable_visible_inframe_corners';return out,info
    try:result=solve(x,original[:8],K,usable)
    except cv2.error:result=None
    if result is None:info['reason']='visible_only_PnP_failed';return out,info
    R2,t2,residual=result
    if not np.isfinite(R2).all() or not np.isfinite(t2).all() or np.min(((R2@x.T).T+t2)[:,2])<=0:
        info['reason']='invalid_pose';return out,info
    hidden2,_,_=visibility(x,R2,t2)
    if not np.array_equal(hidden,hidden2):info['reason']='hidden_set_changed_after_refit';return out,info
    rvec=cv2.Rodrigues(R2)[0]
    projected=cv2.projectPoints(x,rvec,t2,K,None)[0].reshape(-1,2)
    if not np.isfinite(projected).all():info['reason']='invalid_projection';return out,info
    out[np.flatnonzero(hidden)]=projected[hidden]
    np.testing.assert_array_equal(out[np.r_[~hidden,True]],original[np.r_[~hidden,True]])
    info.update(applied=True,reason='replaced_hidden_only',visible_fit_mean_px=float(residual),
                refit_R=R2.tolist(),refit_t=t2.tolist(),projected=projected.tolist())
    return out,info


def tests():
    x=cuboid(1.1,.11,1.3);R=np.eye(3)
    hidden,_,_=visibility(cuboid(1.1,1.1,1.3),R,np.array([0.,0.,3.]))
    assert np.flatnonzero(hidden).tolist()==[4,5,6,7]
    C=np.array([2.,-2.,-3.]);hidden,_,_=visibility(x,R,-C)
    assert np.flatnonzero(hidden).tolist()==[7]
    K=np.array([[600.,0.,320.],[0.,600.,240.],[0.,0.,1.]])
    # Look at a cuboid from above/front/right; one hidden corner is corrupted.
    rvec=np.array([.5,.4,0.]);R=cv2.Rodrigues(rvec)[0];t=np.array([0.,0.,3.])
    points=cv2.projectPoints(x,rvec,t,K,None)[0].reshape(-1,2)
    hidden,_,_=visibility(x,R,t);assert hidden.sum()==1
    q=np.vstack([points,[[320.,240.]]]);q[np.flatnonzero(hidden)]+=[30.,-20.]
    candidate=dict(keypoints_xy=q.tolist(),keypoints_conf=[1.]*9)
    p,info=correct(candidate,dict(available=True,cf_extents=[1.1,.11,1.3],R_cf=R.tolist(),centroid=t.tolist()),K)
    assert info['applied'],info
    np.testing.assert_allclose(p[:8],points,atol=1e-5)
    np.testing.assert_array_equal(p[np.r_[~hidden,True]],q[np.r_[~hidden,True]])
    assert not set(info['used'])&set(info['hidden'])
    return dict(front_view_hidden_far4=True,oblique_hidden_corner=True,synthetic_hidden_error_recovered=True,visible_and_center_exact=True)


def svg(image,before,after,gt,valid,hidden,view='0 0 640 480'):
    def lines(points,color):
        return ''.join(f'<line x1="{points[a][0]}" y1="{points[a][1]}" x2="{points[b][0]}" y2="{points[b][1]}" stroke="{color}" stroke-width="1"/>' for a,b in G.EDGES)
    content=f'<image href="{image}" width="640" height="480"/>'+lines(gt,'#54fa62')+lines(after,'#36baff')
    for j in range(8):
        x,y=after[j]
        if j in hidden:
            bx,by=before[j]
            content+=f'<circle cx="{bx}" cy="{by}" r="4" stroke="#ffad42" fill="none"/><line x1="{bx}" y1="{by}" x2="{x}" y2="{y}" stroke="white" stroke-width="2"/>'
        content+=f'<circle cx="{x}" cy="{y}" r="2" fill="#36baff"/><text x="{x+4}" y="{y-4}" fill="#36baff" stroke="#142029" stroke-width=".4" font-size="12">P{j}</text>'
        if valid[j]:
            gx,gy=gt[j];content+=f'<path d="M {gx-3} {gy} h 6 M {gx} {gy-3} v 6" stroke="#54fa62"/>'
    return f'<svg viewBox="{view}" xmlns="http://www.w3.org/2000/svg">{content}</svg>'


def main():
    cv2.setNumThreads(1);OUT.mkdir(exist_ok=True)
    test=tests()
    paths=[BASE/p for p in ['INPUTS.json','CACHED_PREDICTIONS.json','REPLAY_PREDICTIONS.json','POSE_PREDICTIONS.json','filter_audit/DECISIONS.json']]
    paths += [Path(__file__),Path(F.__file__),ROOT/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py']
    sources=[G.binding(p) for p in paths]
    G.write(OUT/'PROTOCOL.json',dict(arms=ARMS,frames=18,method='frozen initial predicted pose; convex cuboid incident-face visibility; visible-only SQPnP+LM; replace confident hidden only',
        angle_margin_deg=2,min_visible_inframe_confident_corners=6,kp_confidence=.5,hidden_set_must_be_stable=True,
        dimensions='frozen initial prediction-only W/D selector; no new GT axis selection',
        no_GT_for_inference=True,no_training=True,no_threshold_search=True,no_new_detector_inference=True,
        caveat='Solid cuboid proxy, not actual pallet mesh visibility; does not detect external occlusion. Reused CAD DEV, not independent confirmation.',sources=sources,tests=test))
    inputs=G.read(paths[0]);pred=G.read(paths[1]);pred.update(G.read(paths[2])['predictions'])
    poses=G.read(paths[3]);audit=G.read(paths[4]);meta={r['id']:r for r in audit['rows']}
    result={};rows=[]
    for arm in ARMS:
        result[arm]={}
        for r in inputs:
            fid=r['id'];p=copy.deepcopy(pred[arm][fid]);idx=p['selected_index'];c=p['candidates'][idx]
            q,info=correct(c,poses[arm][fid],np.array(meta[fid]['K']))
            c['keypoints_xy']=q.tolist();result[arm][fid]=p
            expected=copy.deepcopy(pred[arm][fid]);expected['candidates'][idx]['keypoints_xy']=q.tolist();assert p==expected
            info.update(id=fid,arm=arm,after=q.tolist())
            rows.append(info)
    G.write(OUT/'PREDICTIONS.json',dict(GT_input=False,predictions=result,decisions=rows))
    # Ground truth is first used below, after all predictions/decisions are frozen.
    metrics=G.read(BASE/'METRICS.json');summary={};evaluated=[]
    for arm in ARMS:
        before_all=[];after_all=[];deltas=[];hidden_before=[];hidden_after=[];applied=0;filtered=0
        for info,m in zip([r for r in rows if r['arm']==arm],metrics['rows'][arm]):
            assert info['id']==m['id'];q=np.array(info['after']);gt=np.array(m['gt']);valid=np.array(m['valid']);valid[8]=False
            e0=np.array(m['errors_px']);e1=np.linalg.norm(q-gt,axis=1);before_all.extend(e0[valid]);after_all.extend(e1[valid]);deltas.append(float(e1[valid].mean()-e0[valid].mean()))
            moved=[j for j in info['hidden'] if info['applied'] and valid[j]]
            hidden_before.extend(e0[moved]);hidden_after.extend(e1[moved]);applied+=int(info['applied'])
            K=np.array(meta[info['id']]['K']);s=F.geometry_scores(q,np.isfinite(q).all(1),K,dict(x=1.1,y=.11,z=1.3))
            passes=meta[info['id']]['stage1_pass'] and s['s_remove']<=.05
            filtered+=int(not passes)
            evaluated.append(dict(**info,gt=gt.tolist(),valid=valid.tolist(),before_errors=e0.tolist(),after_errors=e1.tolist(),LOO=s['s_remove'],filter_pass=passes))
        def stats(e):
            e=np.array(e);return dict(median_px=float(np.median(e)),P90_px=float(np.quantile(e,.9)),PCK20=float(np.mean(e<=20))) if len(e) else None
        summary[arm]=dict(before=stats(before_all),after=stats(after_all),applied_frames=applied,
            improved_frames=sum(d < -1e-8 for d in deltas),worsened_frames=sum(d > 1e-8 for d in deltas),unchanged_frames=sum(abs(d)<=1e-8 for d in deltas),
            replaced_supervised_corners=len(hidden_before),replaced_before=stats(hidden_before),replaced_after=stats(hidden_after),filter_rejected=filtered)
    G.write(OUT/'METRICS.json',dict(summary=summary,rows=evaluated,metric='Fixed-index supervised corner8, same CAD branch0 as prior symmetry matching',
        caveat='Existing hidden-corner annotation may be PnP-derived. Not independent evidence of physical hidden-corner accuracy.'))
    table=''
    for arm,s in summary.items():
        b,a=s['before'],s['after']
        table+=f'<tr><td>{arm}</td><td>{s["applied_frames"]}/18</td><td>{b["median_px"]:.2f} → {a["median_px"]:.2f}</td><td>{b["P90_px"]:.2f} → {a["P90_px"]:.2f}</td><td>{100*b["PCK20"]:.1f}% → {100*a["PCK20"]:.1f}%</td><td>{s["improved_frames"]}/{s["worsened_frames"]}/{s["unchanged_frames"]}</td></tr>'
    cards=[]
    for i,r in enumerate(inputs,1):
        panels=[];image=os.path.relpath(ROOT/r['image']['path'],OUT);G.verify(r['image'])
        for arm in ARMS:
            row=next(x for x in evaluated if x['id']==r['id'] and x['arm']==arm)
            q0,q1,gt=row['before'],row['after'],row['gt'];h=row['hidden'];parts=[]
            for j in h:
                parts.append(f'P{j}: {row["before_errors"][j]:.2f} → {row["after_errors"][j]:.2f}px'+(' (평가 제외)' if not row['valid'][j] else ''))
            panel=svg(image,q0,q1,gt,row['valid'],h)
            zoom=''
            for j in h:
                xy=(np.array(q0[j])+q1[j])/2;zoom+=svg(image,q0,q1,gt,row['valid'],h,f'{xy[0]-65} {xy[1]-65} 130 130')
            panels.append(f'<article><h3>{arm}</h3><p>추정 가림: {h} · {"대체" if row["applied"] else "유지"}</p><small>{row["reason"]} · PnP 입력 {row["used"]}</small>{panel}<div class="zoom">{zoom}</div><p>{"<br>".join(parts)}</p><p>대체 후 LOO {row["LOO"]:.5f} · {"통과" if row["filter_pass"] else "탈락"}</p></article>')
        cards.append(f'<section class="card" id="frame-{i}"><h2>{i:02}/18 · {html.escape(r["id"])}</h2><div class="panels">{"".join(panels)}</div></section>')
    page=f'''<!doctype html><meta charset="utf-8"><title>CAD18 자기 가림 점만 PnP 대체</title><style>body{{background:#101c22;color:#eef5f6;font:16px system-ui;margin:20px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #62727a;text-align:left}}.panels{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}}svg{{width:100%;background:#16232b}}.zoom{{display:flex}}.zoom svg{{max-width:220px}}section{{border-top:3px solid #73838b;margin-top:30px}}small{{display:block;min-height:44px}}a{{color:#70d2ff}}</style><h1>자기 가림 코너만 PnP 대체 · CAD18</h1><p>주황 원: 대체 전 · 파랑: 대체 후 · 흰 선: 실제 이동 · 초록: 기존 GT. 보이는 점과 중심점은 그대로 유지.</p><p>초기 예측 자세 + 치수 직육면체로 가림을 추정. 보이는 고신뢰·화면 내부 점 6개 이상으로만 재계산. 가림 집합이 바뀌면 대체하지 않음. GT로 가림을 정하지 않음.</p><p>직육면체 근사여서 실제 구멍/틈과 외부 가림은 표현하지 못합니다. 가려진 GT 일부가 PnP 기반일 수 있어 물리적 정확도 독립 검증은 아닙니다. 추가 학습 없음.</p><table><tr><th>방법</th><th>대체한 프레임</th><th>코너 중앙값 px</th><th>P90 px</th><th>PCK20</th><th>개선/악화/유지</th></tr>{table}</table><p><a href="#frame-5">질문한 5번</a></p>{''.join(cards)}'''
    G.write(OUT/'index.html',page)
    for b in sources:G.verify(b)
    print(json.dumps(dict(summary=summary,tests=test),indent=2))


if __name__=='__main__':main()
