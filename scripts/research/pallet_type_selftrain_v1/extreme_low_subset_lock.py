"""Lock RGB-only review decisions before reading model metrics (2026-09-20)."""
from collections import Counter
from html import escape
from . import common as C
from . import elevation_review as V
from .extreme_low_subset import DOC, OUT

# Global indices in the hash-bound, session/id-ordered FRAMES.json.
# All 17 contact sheets reviewed; borderline cases retained deliberately.
EXCLUDE = {64, 65, 66, *range(97, 128), 192}
BORDERLINE_KEEP = {*range(18, 30), *range(37, 42), 63, 67, 95, 96, 156, 182}


def main():
    protocol = C.read(DOC/'PROTOCOL.json')
    for b in protocol['sources']:
        C.verify(b)
    frames = C.read(V.OUT/'FRAMES.json')
    assert len(frames) == 194 and len(EXCLUDE) == 35
    decisions = []
    for i, r in enumerate(frames):
        C.verify(r['image'])
        if i in EXCLUDE:
            reason = 'RGB: upper face nearly edge-on; far/near upper boundaries and corners not reliably separable.'
        elif i in BORDERLINE_KEEP:
            reason = 'Borderline retained: narrow but visible upper-face strip/grid, or insufficient certainty that elevation alone prevents corner discrimination.'
        else:
            reason = 'Keep: upper-face extent visible, or no clear extreme-low evidence. Darkness/occlusion/truncation alone is not an exclusion.'
        decisions.append(dict(index=i, **r, decision='exclude' if i in EXCLUDE else 'keep', reason=reason))
    original = C.read(C.DOC/'EVAL_PROTOCOL.json')
    records = original['records']
    assert len(records) == 469
    by_id = {r['id']: r for r in records}
    assert len(by_id) == 469
    assert {r['id'] for r in records if r['kind'] == 'PLASTIC'} == {r['id'] for r in frames}
    for r in records:
        C.verify(r['image'])
        C.verify(r['annotation'])
    excluded = {r['id'] for r in decisions if r['decision'] == 'exclude'}
    kept = [r for r in records if r['id'] not in excluded]
    manifest = dict(
        status='FROZEN_POSTHOC_RGB_SCOPE_SUBSET',
        name='ordinary_plastic_without_clear_extreme_low_v1',
        counts=dict(Counter(r['kind'] for r in kept)),
        reviewed=194, excluded_count=len(excluded), retained_count=len(kept),
        excluded_by_session=dict(Counter(r['session'] for r in decisions if r['decision']=='exclude')),
        sources=[C.bound(DOC/'PROTOCOL.json'), C.bound(C.DOC/'EVAL_PROTOCOL.json'), C.bound(__file__)],
        review_method='Assistant RGB-only qualitative review; all194 frames reviewed on sheets00-16, selected boundary cases opened at original resolution. No GT/prediction/error overlays used. Not a numerical camera-angle threshold.',
        limitations='Posthoc scope restriction of a development-exposed evaluation, not a new independent test. Full194 must remain reported. Exclusion is not model improvement or recovery.',
        training_prohibited_ids=[r['id'] for r in records],
        originals_preserved=True, historical_evaluation_defaults_unchanged=True,
        decisions=decisions, records=kept)
    C.freeze(DOC/'SUBSET_MANIFEST.json',manifest)
    cards=[]
    for r in decisions:
        if r['decision']!='exclude':continue
        src='../elevation_review/'+r['preview']
        cards.append(f'<figure><figcaption>{r["index"]:03d} {escape(r["id"])}</figcaption><img loading="lazy" src="{escape(src)}"><p>{escape(r["reason"])}</p></figure>')
    html='<!doctype html><html lang="ko"><meta charset="utf-8"><title>극저각 제외 35장</title><style>body{background:#132028;color:#eee;font:16px sans-serif;margin:24px}figure{margin:20px 0}img{width:640px;max-width:100%}p{max-width:900px}</style><h1>극저각 제외 35장 · 일반 플라스틱 194 → 159장</h1><p>원본 RGB만 보고 정성 판정. 애매한 장면은 유지. 초록150·목재125·negative는 그대로. 원본/주석/전체194 결과는 보존하며, 제외 사진도 학습에 사용하지 않습니다. 평가 범위의 사후 변경이지 모델 개선이 아닙니다.</p>'+''.join(cards)+'</html>'
    C.write_text(OUT/'excluded.html',html)
    C.freeze(DOC/'LOCK_COMPLETE.json',dict(manifest=C.bound(DOC/'SUBSET_MANIFEST.json'), gallery=C.bound(OUT/'excluded.html'), model_metrics_read_for_selection=False, original_annotations_verified=469))
    print(manifest['counts'],manifest['excluded_by_session'])


if __name__=='__main__':main()
