"""Prepare disjoint real annotation proposals, never labels or training data.

Prediction uncertainty only prioritizes manual review; it is not correctness.
Existing annotations, releases and pseudo-label decisions remain untouched.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import common as C
from .pseudo import top

DOC = C.DOC / 'large_corner_recovery_v1/real_support_review'
OUT = C.OUT / 'large_corner_recovery_v1/real_support_review'
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
# Metadata-only proposal; these are not committed train/eval assignments.
QUOTAS = {
    'capturepallet11': ('proposed_support', 10),
    'capturenight01': ('proposed_support', 3),
    'capturenight03': ('proposed_support', 3),
    'capturenight02': ('proposed_support', 2),
    'capturenight04': ('proposed_support', 2),
    'capturepallet10': ('proposed_validation', 5),
    'capturenight10': ('proposed_validation', 5),
}


def closure(seeds, groups):
    """Conservative closure over aliases and partial recording overlaps."""
    seen = set(seeds)
    edges = [set(s['session_key'] for s in g['sessions']) for g in groups['groups'] if not g['is_collection']]
    edges += [{p['session_a'], p['session_b']} for p in groups['partial_overlap_pairs']]
    while True:
        old = len(seen)
        for edge in edges:
            if edge & seen:
                seen |= edge
        if old == len(seen):
            return seen


def uncertainty(frame):
    p = top(frame['raw'])
    if p is None:
        return 1.0
    values = [1 - float(p['score']), 1 - float(np.mean(p['keypoints_conf'][:8]))]
    # Existing geometric scores are review signals, NOT a new rejection rule.
    for scores in [frame.get('stage1'), frame.get('stage2')]:
        for value in (scores or {}).values():
            if value is not None and np.isfinite(value):
                values.append(min(float(value), 1.0))
    if frame.get('refined') and top(frame['refined']) is not None:
        q = top(frame['refined'])
        delta = np.asarray(q['keypoints_xy'])[:8] - np.asarray(p['keypoints_xy'])[:8]
        diag = max(float(np.linalg.norm(np.asarray(p['box_xyxy'])[2:] - p['box_xyxy'][:2])), 1.)
        if np.isfinite(delta).all():
            values.append(min(float(np.linalg.norm(delta, axis=1).max()) / diag, 1.))
    return max(values)


def select_spread(rows, count, validation=False):
    """One per contiguous time bin; validation selection ignores scores."""
    ordered = sorted(rows, key=lambda r: (Path(r['image']['path']).stem, r['id']))
    assert len(ordered) >= count
    selected = []
    for i, indices in enumerate(np.array_split(np.arange(len(ordered)), count)):
        part = [ordered[int(j)] for j in indices]
        if validation:
            row = part[len(part)//2]
            reason = 'time_bin_midpoint_no_model_ranking'
        elif i % 3 == 2:
            row = min(part, key=lambda r: (r['review_uncertainty'], r['id']))
            reason = 'lower_uncertainty_control_NOT_verified_correct'
        else:
            row = min(part, key=lambda r: (-r['review_uncertainty'], r['id']))
            reason = 'higher_uncertainty_NOT_verified_wrong'
        selected.append(dict(row, selection_reason=reason, temporal_bin=i))
    return selected


def overlay(image, prediction):
    canvas = image.copy()
    p = top(prediction)
    if p is None:
        cv2.putText(canvas, 'NO R0 DETECTION', (15,35), 0, .8, (0,0,255), 2)
        return canvas
    q = np.asarray(p['keypoints_xy'])
    for a,b in EDGES:
        if np.isfinite(q[[a,b]]).all():
            cv2.line(canvas, tuple(np.rint(q[a]).astype(int)), tuple(np.rint(q[b]).astype(int)), (0,220,255), 1, cv2.LINE_AA)
    for i, xy in enumerate(q[:8]):
        if not np.isfinite(xy).all():
            continue
        pt = tuple(np.rint(xy).astype(int))
        cv2.circle(canvas, pt, 4, (0,220,255), 2)
        loc = (max(0,min(canvas.shape[1]-30,pt[0]+5)), max(15,min(canvas.shape[0]-5,pt[1]-5)))
        cv2.putText(canvas, f'P{i}', loc, 0, .5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, f'P{i}', loc, 0, .5, (0,220,255), 1, cv2.LINE_AA)
    return canvas


def main():
    pool_path = C.DOC/'POOL.json'
    pool = C.read(pool_path)
    for binding in pool['sources']:
        C.verify(binding)
    groups_path = C.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json'
    groups = C.read(groups_path)
    dev_path = C.DOC/'EVAL_PROTOCOL.json'
    dev = C.read(dev_path)['records']
    evalhash = {r['image']['sha256'] for r in dev}
    evalsessions = {str(Path(r['image']['path']).parent.parent) for r in dev}
    evalsessions |= {s['session_key'] for g in groups['groups'] if g['recording_id'] in pool['evaluation_recording_ids'] for s in g['sessions']}
    forbidden = closure(evalsessions, groups)
    population = [r for r in pool['records'] if r['kind']=='PLASTIC']
    by_session = defaultdict(list)
    frame_sources = []
    for r in population:
        assert r['session'] not in forbidden and r['image']['sha256'] not in evalhash
        path = C.RAW/'pseudo_frames'/(r['id']+'.json')
        frame = C.read(path)
        assert frame['image'] == r['image'] and frame['id'] == r['id']
        assert frame['protocol_sha256'] == C.sha(C.DOC/'PSEUDO_PROTOCOL.json')
        frame_sources.append(C.bound(path))
        by_session[Path(r['session']).name].append(dict(r, review_uncertainty=uncertainty(frame),
            old_filter_reason=frame['reason'], old_filter_accepted=frame['accepted'],
            prediction_source=C.bound(path)))
    selected = []
    for session, (role, count) in QUOTAS.items():
        selected += [dict(r, proposed_role=role) for r in select_spread(by_session[session], count, role=='proposed_validation')]
    support = {r['session'] for r in selected if r['proposed_role']=='proposed_support'}
    val = {r['session'] for r in selected if r['proposed_role']=='proposed_validation'}
    assert not closure(support, groups) & closure(val, groups)
    assert len({r['image']['sha256'] for r in selected}) == 30
    assert Counter(r['proposed_role'] for r in selected) == {'proposed_support':20,'proposed_validation':10}
    legacy_counts = {}
    for session in QUOTAS:
        legacy_counts[session] = len(list((C.ROOT/'challenge/data/01_real/manual_gt'/f'{session}_manual_gt').glob('*.json')))
    protocol = dict(status='PROPOSAL_ONLY_REQUIRES_HUMAN_CORNER_REVIEW', population=1000, quotas=QUOTAS,
        selection='Per-session contiguous timestamp bins; support high uncertainty except every third bin low uncertainty; validation midpoint without uncertainty ranking.',
        uncertainty='Max of 1-boxconfidence,1-meankpconfidence,existing flip/LOO scores,normalized refiner max displacement; review priority only, no correctness claim.',
        exposure='All1000 have historical teacher/pseudo experiments; proposed validation is only disjoint from proposed support, NOT a globally untouched test. Final claims require independent confirmation.',
        labels='No coordinates generated, no original annotation edits, no pseudo/filter changes, no train/eval reassignment, no training. Legacy manual_kps may include projected points and must not be automatically treated as manual supervision.',
        evaluation_scope='Existing per-type EVAL_PROTOCOL plus POOL DEV+GREEN150 recording exclusions, transitive aliases and partial overlaps.',
        original_model_unchanged=True, existing_eval_unchanged=True, GT_used_for_selection=False,
        code=C.bound(__file__), sources=[C.bound(pool_path),C.bound(dev_path),C.bound(groups_path),C.bound(C.DOC/'PSEUDO_PROTOCOL.json')],
        candidate_prediction_sources=frame_sources)
    C.freeze(DOC/'PROTOCOL.json',protocol)
    cards = []
    (OUT/'images').mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(selected):
        C.verify(r['image']); C.verify(r['prediction_source'])
        frame = C.read(C.ROOT/r['prediction_source']['path'])
        image = cv2.imread(str(C.ROOT/r['image']['path'])); assert image is not None
        panel = np.concatenate([image, overlay(image,frame['raw'])],axis=1)
        dest = OUT/'images'/f'{i:02d}.jpg'
        if not dest.exists():
            assert cv2.imwrite(str(dest),panel,[cv2.IMWRITE_JPEG_QUALITY,94])
        legacy = C.ROOT/'challenge/data/01_real/manual_gt'/f"{Path(r['session']).name}_manual_gt"/(Path(r['image']['path']).stem+'.json')
        cards.append(dict(r,preview=f'images/{i:02d}.jpg',preview_binding=C.bound(dest),
            existing_legacy_annotation=C.bound(legacy) if legacy.exists() else None,
            annotation_status='NOT_VERIFIED', corners_verified=0))
    report = dict(complete=True,proposal_only=True,labels_created=0,models_trained=0,
        pool=1000,selected=30,support_candidates=20,validation_candidates=10,
        support_recordings=sorted({r['recording_id'] for r in selected if r['proposed_role']=='proposed_support'}),
        validation_recordings=sorted({r['recording_id'] for r in selected if r['proposed_role']=='proposed_validation'}),
        eval_recording_overlap=False,support_validation_recording_overlap=False,duplicate_hashes=0,
        old_filter_reasons=dict(Counter(r['old_filter_reason'] for r in selected)),
        legacy_annotation_counts_in_selected_sessions=legacy_counts,
        legacy_annotations_not_used_as_GT=True,source=C.bound(DOC/'PROTOCOL.json'))
    C.freeze(OUT/'CANDIDATES.json',cards)
    C.freeze(DOC/'AUDIT.json',report)
    payload=json.dumps(cards,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>실사 보정 학습 후보30 — 아직 정답 아님</title>
<style>body{background:#101820;color:#edf2f7;font:17px sans-serif;max-width:1300px;margin:24px auto;padding:16px}article{border:1px solid #567;margin:24px 0;padding:15px}img{width:100%;height:auto}select{font:inherit;padding:10px}.warning{background:#513329;padding:15px}code{overflow-wrap:anywhere}small{color:#bfd0df}</style>
<h1>일반 플라스틱: 실사 정답 확인 후보30장</h1>
<p class="warning">학습 제안20장 + 별도 촬영 세션 검증 제안10장. 아직 학습 데이터가 아닙니다. 정답0장 생성, 재학습0회. 기존 평가·수도레이블·최종모델은 변경하지 않았습니다.</p>
<p>왼쪽 원본 / 오른쪽 기존 R0 예측(P0–P7). 정답이나 보정 결과가 아닙니다. 불확실도가 크다는 이유만으로 틀렸다고 판정하지 않습니다. 낮아도 맞는다는 보장은 없습니다.</p>
<p>학습 후보는 시간 구간마다 확인 우선순위가 높은 예측과 낮은 예측을 섞었습니다. 검증 후보는 예측 점수와 무관하게 시간 구간 중앙에서 골랐습니다. 전체 풀은 과거 연구에서 이미 사용되어, 완전히 새로운 독립 시험은 아닙니다.</p>
<p>확인할 항목: 실제 팔레트 종류, 코너 번호, 보이는 코너의 직접 클릭 좌표. 가려진 점·추정점·PnP 투영점을 직접 관측 정답으로 사용하지 않습니다. 기존 legacy JSON의 manual_kps도 검증 없이 재사용하지 않습니다.</p>
<select id="role"><option value="all">전체30장</option><option value="proposed_support">학습 제안20장</option><option value="proposed_validation">검증 제안10장</option></select><p id="count"></p><main id="cards"></main>
<script>const DATA=__DATA__;function render(){const rows=DATA.filter(r=>document.querySelector('#role').value==='all'||r.proposed_role===document.querySelector('#role').value);document.querySelector('#count').textContent=rows.length+'장';const root=document.querySelector('#cards');root.replaceChildren();for(const r of rows){const a=document.createElement('article'),h=document.createElement('h3'),p=document.createElement('p'),im=document.createElement('img'),path=document.createElement('code');h.textContent=(r.proposed_role==='proposed_support'?'학습 제안':'검증 제안')+' | '+r.session.split('/').pop()+' | '+r.image.path.split('/').pop();p.textContent='기존 필터: '+r.old_filter_reason+' | 확인 우선순위: '+r.review_uncertainty.toFixed(3)+' | '+r.selection_reason;im.src=r.preview;im.loading='lazy';im.alt=h.textContent;path.textContent=r.image.path;a.append(h,p,im,path);root.append(a)}}document.querySelector('#role').addEventListener('change',render);render();</script></html>'''.replace('__DATA__',payload)
    C.write_text(OUT/'index.html',page)
    print(json.dumps(report,ensure_ascii=False,indent=2));print(OUT/'index.html')


if __name__=='__main__':
    main()
