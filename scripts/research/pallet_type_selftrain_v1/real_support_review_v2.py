"""Correct V1 background-biased review sampling; no training/GT changes."""
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
from . import real_support_review as V

C=V.C
DOC=C.DOC/'large_corner_recovery_v1/real_support_review_v2'
OUT=C.OUT/'large_corner_recovery_v1/real_support_review_v2'


def eligible(pred):
    p=V.top(pred)
    if p is None or p['score']<.5:
        return False
    box=np.array(p['box_xyxy'],float);q=np.array(p['keypoints_xy'],float)[:8]
    return bool(np.isfinite(box).all() and np.isfinite(q).all() and
                not (q==-1).all(1).any() and np.all(box[2:]>box[:2]) and np.linalg.norm(box[2:]-box[:2])>=120)


def main():
    old=C.read(V.DOC/'PROTOCOL.json')
    for b in old['sources']+old['candidate_prediction_sources']:
        C.verify(b)
    C.verify(old['code'])
    pool=C.read(C.DOC/'POOL.json');groups=C.read(C.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json')
    dev=C.read(C.DOC/'EVAL_PROTOCOL.json')['records'];eh={r['image']['sha256'] for r in dev}
    forbidden=V.closure({str(Path(r['image']['path']).parent.parent) for r in dev} |
        {s['session_key'] for g in groups['groups'] if g['recording_id'] in pool['evaluation_recording_ids'] for s in g['sessions']},groups)
    by=defaultdict(list);allrows=[r for r in pool['records'] if r['kind']=='PLASTIC']
    for r in allrows:
        assert r['session'] not in forbidden and r['image']['sha256'] not in eh
        fp=C.RAW/'pseudo_frames'/(r['id']+'.json');f=C.read(fp)
        if not eligible(f['raw']):
            continue
        by[Path(r['session']).name].append(dict(r,review_uncertainty=V.uncertainty(f),
            old_filter_reason=f['reason'],prediction_source=C.bound(fp)))
    selected=[]
    for s,(role,n) in V.QUOTAS.items():
        selected.extend(dict(r,proposed_role=role) for r in V.select_spread(by[s],n,role=='proposed_validation'))
    a={r['session'] for r in selected if r['proposed_role']=='proposed_support'}
    b={r['session'] for r in selected if r['proposed_role']=='proposed_validation'}
    assert not V.closure(a,groups)&V.closure(b,groups)
    assert len(selected)==len({r['image']['sha256'] for r in selected})==30
    protocol=dict(status='PROPOSAL_ONLY_NOT_TRAINING_LABELS',
        correction='V1 visual inspection found background/no-detection examples. Preserve V1; restrict new review to detected candidate support. No evaluation errors/GT used.',
        eligibility='R0 detection score>=0.5, finite8points/no sentinel, positive-size predicted bbox diagonal>=120 raw px. This is annotation eligibility only, not a changed pseudo-label/evaluation filter and not verified object presence.',
        quotas=V.QUOTAS,selection=old['selection'],uncertainty=old['uncertainty'],
        exposure=old['exposure'],labels=old['labels'],evaluation_scope=old['evaluation_scope'],
        sources=[C.bound(V.DOC/'PROTOCOL.json'),C.bound(V.__file__),C.bound(__file__)],
        source_pool=1000,eligible=sum(map(len,by.values())),eligible_sessions={s:len(r) for s,r in by.items()},
        human_verification_required=True,training_started=False,GT_used_for_selection=False)
    C.freeze(DOC/'PROTOCOL.json',protocol)
    (OUT/'images').mkdir(parents=True,exist_ok=True);cards=[];articles=[]
    for i,r in enumerate(selected):
        C.verify(r['image']);C.verify(r['prediction_source']);f=C.read(C.ROOT/r['prediction_source']['path'])
        im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
        panel=np.concatenate([im,V.overlay(im,f['raw'])],1);dest=OUT/'images'/f'{i:02d}.jpg'
        if not dest.exists():assert cv2.imwrite(str(dest),panel,[cv2.IMWRITE_JPEG_QUALITY,94])
        cards.append(dict(r,preview=f'images/{i:02d}.jpg',preview_binding=C.bound(dest),annotation_status='NOT_VERIFIED',corners_verified=0))
        role='학습 제안' if r['proposed_role']=='proposed_support' else '검증 제안'
        title=f"{i+1:02d}. {role} | {Path(r['session']).name} | {Path(r['image']['path']).stem}"
        articles.append(f'<article data-role="{r["proposed_role"]}"><h3>{html.escape(title)}</h3><p>기존 필터: {r["old_filter_reason"]} | 확인 우선순위 {r["review_uncertainty"]:.3f} | {r["selection_reason"]}</p><img src="images/{i:02d}.jpg" loading="lazy" alt="{html.escape(title)}"><code>{html.escape(r["image"]["path"])}</code></article>')
    C.freeze(OUT/'CANDIDATES.json',cards)
    audit=dict(complete=True,proposal_only=True,pool=1000,eligible=protocol['eligible'],selected=30,
        support_candidates=20,validation_candidates=10,duplicate_images=0,eval_recording_overlap=False,
        support_validation_recording_overlap=False,labels_created=0,models_trained=0,
        old_filter_reasons=dict(Counter(r['old_filter_reason'] for r in cards)),
        source=C.bound(DOC/'PROTOCOL.json'),candidates=C.bound(OUT/'CANDIDATES.json'))
    C.freeze(DOC/'AUDIT.json',audit)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>실사 코너 확인 후보30장 v2</title>
<style>body{background:#101820;color:#edf2f7;font:17px sans-serif;max-width:1300px;margin:24px auto;padding:16px}article{border:1px solid #567;margin:24px 0;padding:15px}img{width:100%;height:auto}select{font:inherit;padding:10px}.warning{background:#513329;padding:15px}code{overflow-wrap:anywhere}</style>
<h1>일반 플라스틱: 실사 코너 확인 후보30장</h1>
<p class="warning">학습 제안20장 + 다른 촬영 세션의 검증 제안10장. 아직 정답이 아닙니다. 코너 라벨 생성0 / 재학습0. 기존 평가·수도레이블·최종모델은 변경하지 않았습니다.</p>
<p>왼쪽 원본 / 오른쪽 기존 R0 예측(P0–P7). 보정 후 결과나 정답이 아닙니다. 가려진 점은 임의로 찍지 말고, 보이는 코너 위치와 번호를 확인해야 합니다.</p>
<p>첫 선별의 배경 편향을 수정했습니다. 기존 풀1000장 중 검출 신뢰도0.5 이상·예측 상자 대각선120px 이상인485장을 대상으로 시간 구간별로 골랐습니다. 이는 어노테이션 후보 조건이며 기존 학습/평가 필터를 바꾸지 않습니다. 오검출이 남을 수 있습니다.</p>
<p>학습 후보는 높은·낮은 불확실도를 섞었고, 검증 후보는 예측 순위 없이 시간 구간 중앙에서 뽑았습니다. 이 풀에는 과거 연구 노출이 있어 새로운 독립 시험이라고 주장하지 않습니다. 보이는 점만 직접 확인하고 legacy 투영 좌표를 수동 정답으로 재사용하지 않습니다.</p>
<select id="role"><option value="all">전체30장</option><option value="proposed_support">학습 제안20장</option><option value="proposed_validation">검증 제안10장</option></select><p id="count">30장</p><main>__CARDS__</main>
<script>document.querySelector('#role').addEventListener('change',e=>{let n=0;for(const a of document.querySelectorAll('article')){a.hidden=e.target.value!=='all'&&a.dataset.role!==e.target.value;if(!a.hidden)n++;}document.querySelector('#count').textContent=n+'장';});</script></html>'''.replace('__CARDS__','\n'.join(articles))
    assert protocol['eligible']==485
    C.write_text(OUT/'index.html',page)
    print(json.dumps(audit,ensure_ascii=False,indent=2));print(OUT/'index.html')


if __name__=='__main__':main()
