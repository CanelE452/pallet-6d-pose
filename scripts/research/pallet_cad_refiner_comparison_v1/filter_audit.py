"""Read-only prediction audit: exact existing two-stage whole-frame filter."""
import sys
from pathlib import Path
import json
import html
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_cad_r0_gt_gallery_v1 import build as G
from scripts.self_training_yolo import pseudo_label_filters as F

BASE=ROOT/'outputs/pallet_cad_refiner_comparison_v1'
OUT=BASE/'filter_audit'
read=G.read
ARMS=['R0','N3_DIM_SYM_seed1','N3_DIM_SYM_seed2','N3_DIM_SYM_seed3','REPLAY_RAW','REPLAY_CAP1PCT']


def top(p):
    return p['candidates'][p['selected_index']]


def valid(c,conf=False):
    q=np.asarray(c['keypoints_xy'])
    v=np.isfinite(q).all(1)&~(q==-1).all(1)
    if conf:v &= np.asarray(c['keypoints_conf'])>=.5
    return v


def main():
    cv2.setNumThreads(1);OUT.mkdir(exist_ok=True)
    cachepath=ROOT/'data/pallet/results/paper_selftrain_v1/M4_FRAME_RECORDS.json'
    lockpath=ROOT/'data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json'
    regpath=ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'
    cache=read(cachepath);lock=read(lockpath)
    assert cache['filter_lock_sha256']==G.binding(lockpath)['sha256']
    assert cache['teacher_sha256']==read(G.PRED)['checkpoint']['sha256']
    assert lock['TAU_BOX']==.85
    assert lock['geometry_thresholds']['tau_remove']==lock['geometry_thresholds']['tau_flip']==.05
    records=[r for r in read(G.PROTOCOL)['records'] if r['session']=='eval_cad']
    old={r['frame_id']:r for r in cache['frames']}
    predictions=read(BASE/'CACHED_PREDICTIONS.json')
    predictions.update(read(BASE/'REPLAY_PREDICTIONS.json')['predictions'])
    dims=next(r['physical_dimensions_m'] for r in read(regpath)['objects'] if r['object_type']=='plastic_standard_110x130x11')
    sources=[BASE/'CACHED_PREDICTIONS.json',BASE/'REPLAY_PREDICTIONS.json',cachepath,lockpath,regpath,Path(__file__),Path(F.__file__)]
    bindings=[G.binding(p) for p in sources]
    rows=[]
    for i,r in enumerate(records,1):
        fid=r['id'];c=top(predictions['R0'][fid]);o=old[fid]
        G.verify(r['image']);G.verify(r['annotation'])
        # Only camera intrinsics are consumed from annotation containers at decision time.
        intr=read(ROOT/r['annotation']['path'])['camera_data']['intrinsics']
        K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],float)
        np.testing.assert_array_equal(c['keypoints_xy'],o['keypoints_xy'])
        np.testing.assert_array_equal(valid(c,True),o['keypoint_valid'])
        assert c['score']==o['box_conf']
        s=F.geometry_scores(np.array(c['keypoints_xy']),valid(c,True),K,dims)
        np.testing.assert_allclose([s['s_remove'],s['s_reproj']],[o['s_remove'],o['s_reproj']],rtol=0,atol=1e-10)
        confidence=c['score']>=.85 and valid(c,True)[:8].sum()>=6
        stage1=bool(confidence and s['s_remove']<=.05 and o['s_flip']<=.05)
        row=dict(index=i,id=fid,K=K.tolist(),camera_source=r['annotation'],raw_confidence_pass=bool(confidence),
                 raw_LOO=s['s_remove'],raw_flip=o['s_flip'],stage1_pass=stage1,arms={})
        for arm in ARMS:
            cand=top(predictions[arm][fid]);q=np.array(cand['keypoints_xy']);v=valid(cand)
            score=F.geometry_scores(q,v,K,dims)
            own=bool(v[:8].all() and np.isfinite(score['s_remove']) and score['s_remove']<=.05)
            hyp=min(score['hypotheses'],key=lambda h:h['s_remove'])
            xyz=dict(F.registry_hypotheses(dims))[hyp['name']]
            loo=[]
            for j in range(8):
                keep=np.array([k for k in range(8) if k!=j]);sol=F._solve(xyz[keep],q[keep],K)
                projected=F._project(xyz[j:j+1],*sol,K)[0]
                loo.append(float(np.linalg.norm(projected-q[j])))
            np.testing.assert_allclose(np.median(loo)/hyp['projected_diagonal_px'],score['s_remove'],atol=1e-12)
            row['arms'][arm]=dict(LOO=score['s_remove'],all8_LOO_pass=own,
                final_pass=stage1 if arm=='R0' else stage1 and own,
                winning_hypothesis=hyp,loo_corner_errors_px=loo,score_details=score)
        rows.append(row)
    summary={arm:dict(total=18,stage1_pass=sum(r['stage1_pass'] for r in rows),
        standalone_all8_LOO_pass=sum(r['arms'][arm]['all8_LOO_pass'] for r in rows),
        final_pass=sum(r['arms'][arm]['final_pass'] for r in rows),
        rejected_indices=[r['index'] for r in rows if not r['arms'][arm]['final_pass']]) for arm in ARMS}
    decision=dict(protocol='R0 confidence .85 / >=6 corners confidence .5 -> raw flip and median LOO .05 -> corrected all8 finite median LOO .05',
        raw_flip='Reused archived R0 flip scores: teacher/filter SHA, all 9 coordinates and validity, box score exact; raw geometry parity <=1e-10.',
        post_refiner_flip=False,GT_coordinates_used_for_decisions=False,predictions_modified=False,
        excluded=['shape','X crossing','extra reprojection threshold','augmentation stability'],
        calibration='session camera intrinsics only',sources=bindings,summary=summary,rows=rows)
    # Freeze decisions before joining GT-based quality scores.
    G.write(OUT/'DECISIONS.json',decision)
    metrics=read(BASE/'METRICS.json');quality={}
    for arm in ARMS:
        rr=metrics['rows'][arm]
        quality[arm]=dict(accepted_bad_max20_indices=[r['index'] for r,m in zip(rows,rr) if r['arms'][arm]['final_pass'] and m['max_px']>20],
            bad_definition='Existing supervised fixed-index corner8+center maximum error >20px. Audit only, never a filter input.')
    G.write(OUT/'QUALITY.json',quality)
    table=''.join(f'<tr><td>{a}</td><td>{s["final_pass"]}/18</td><td>{s["rejected_indices"] or "없음"}</td><td>{s["standalone_all8_LOO_pass"]}/18</td></tr>' for a,s in summary.items())
    cards=[]
    for r in rows:
        i=r['index'];detail=''.join(f'<tr><td>{a}</td><td>{r["arms"][a]["LOO"]:.5f}</td><td>{"통과" if r["arms"][a]["final_pass"] else "탈락"}</td><td>{metrics["rows"][a][i-1]["max_px"]:.2f}px</td></tr>' for a in ARMS)
        cards.append(f'<section class="card" id="frame-{i}"><h2>{i:02}/18 · {html.escape(r["id"])}</h2><p>R0 flip {r["raw_flip"]:.5f} · R0 LOO {r["raw_LOO"]:.5f} · 1차 {"통과" if r["stage1_pass"] else "탈락"}</p><table><tr><th>방법</th><th>LOO (≤0.05)</th><th>최종</th><th>GT 최대 오차 (필터에 미사용)</th></tr>{detail}</table><img src="../{i:02}_comparison.png"><p><a href="../movement/index.html#frame-{i}">보정 이동 화살표 보기</a></p></section>')
    page=f'''<!doctype html><meta charset="utf-8"><title>CAD18 기존 필터 감사</title><style>body{{background:#101c22;color:#eef5f6;font:17px system-ui;max-width:1350px;margin:auto;padding:24px}}a{{color:#79d9ff}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #52616a;text-align:left}}section{{border-top:3px solid #70818a;margin-top:40px;padding-top:10px}}img{{width:100%;height:auto}}</style><h1>CAD 18장 · R0 → 보정 → 기존 필터</h1><p>좌표 수정·새 학습 없음. GT는 오차 표시용으로만 사용. flip은 R0에서 한 번만 검사. 보정 후는 all8 median LOO. 형태/X자/추가 필터 제외.</p><p>LOO 0.05는 픽셀이 아니라 PnP 투영 팔레트 대각선 대비 비율입니다. 통과는 정답 보장이 아닙니다. 과거 flip 캐시는 원본 좌표·confidence·teacher/filter SHA 및 geometry 재계산 일치 확인 후 재사용했습니다.</p><table><tr><th>방법</th><th>최종 통과</th><th>탈락 이미지 번호</th><th>단독 all8 LOO 통과</th></tr>{table}</table><p><a href="#frame-5">질문한 5번 이미지</a></p>{''.join(cards)}'''
    G.write(OUT/'index.html',page)
    for b in bindings:G.verify(b)
    print(json.dumps(dict(summary=summary,frame5=rows[4],quality=quality),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
