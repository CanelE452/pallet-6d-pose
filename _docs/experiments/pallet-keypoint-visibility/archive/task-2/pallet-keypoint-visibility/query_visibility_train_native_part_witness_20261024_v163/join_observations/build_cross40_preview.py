from pathlib import Path
import html
import json

V = Path(__file__).resolve().parents[1]
O = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (O / 'CROSS40_FOUR_WAY_WITNESSES.jsonl').read_text().splitlines()]
summary = json.loads((O / 'CROSS40_AGREEMENT_BEFORE_HISTORICAL_LABEL_JOIN.json').read_text())
lookup = {r['case_id']: r for r in rows}
names = {'VISIBLE_SUPPORTED':'부위 보임', 'EXTERNAL_BLOCK_SUPPORTED':'외부 차폐 지지', 'SELF_HIDE_SUPPORTED':'자기 차폐 지지', 'BOUNDARY_OR_GAP_AMBIGUOUS':'경계·개구 모호', 'UNKNOWN':'미확정', 'BACKGROUND_OR_GAP':'배경·개구', 'ON_INTENDED_PART_SUPPORTED':'의도한 부위 위', 'NEAR_INTENDED_PART_SAME_SELECTED_PALLET':'같은 부위 인접면', 'OTHER_SELECTED_PALLET_PART':'대상 팔레트의 다른 부위', 'FOREIGN_OBJECT':'다른 물체'}
esc = lambda s: html.escape(str(s))
def decision(r, field):
    c = r['consensus'][field]
    return names.get(c['value'], c['value']) if c['unanimous'] else '판독 불일치 → 미확정 유지'

cards = []
explanations = {
    'Q1608': '부위는 보이는데 중심은 배경: 목재 유무만 읽으면 가림과 위치 빗나감을 혼동할 수 있습니다.',
    'Q0094': '같은 팔레트의 목재여도 다른 부위: 하부 러너 끝을 평가하는 query가 상부 데크판에 놓입니다.',
    'Q1213': '외부 가림과 query 물체는 별도: 의도한 끝은 눕힌 통이 가리고, query는 뒤의 금속 통을 가리킵니다.',
    'Q1026': '불일치 사례도 보존: 블록 아래 얇은 목재가 러너 끝을 충분히 입증하는지 판독이 갈립니다.',
}
for cid, description in explanations.items():
    r = lookup[cid]
    card = V / 'evidence_v1/cases' / cid / 'CARD.png'
    votes = ''.join('<tr><td>'+w+'</td><td>'+esc(names[x['native_part_visibility']])+'</td><td>'+esc(names[x['query_relation']])+'</td><td>'+esc(x['reason_KO'])+'</td></tr>' for w,x in r['original_votes'].items())
    cards.append(f'<article id="{cid}"><h2>{cid} · native{r["native_slot"]}</h2><p>{esc(description)}</p><img src="{card}" alt="{cid} 원본과 고정 query crop"><table><thead><tr><th>판독</th><th>부위 가시성</th><th>query 관계</th><th>실제 영상 근거</th></tr></thead><tbody>{votes}</tbody></table></article>')
table = ''.join(f'<tr><td><a href="{V / "evidence_v1/cases" / r["case_id"] / "CARD.png"}">{r["case_id"]}</a></td><td>{r["native_slot"]}</td><td>{esc(decision(r,"native_part_visibility"))}</td><td>{esc(decision(r,"query_relation"))}</td></tr>' for r in rows)
document = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V163 공개 TRAIN · 40사례 판독</title>
<style>body{{font-family:"Noto Sans CJK KR",sans-serif;margin:28px auto;padding:0 24px;max-width:1400px;color:#17242d;background:#f7f9fb;line-height:1.65}}h1{{line-height:1.3}}.notice,article{{background:white;border:1px solid #d8e0e6;border-radius:12px;padding:22px;margin:22px 0}}.notice{{border-left:6px solid #168888}}img{{display:block;width:100%;height:auto;margin:15px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #dce3e9;text-align:left;vertical-align:top}}th{{background:#eef3f7}}a{{color:#075ea8}}.small{{font-size:13px;color:#495b68}}article{{scroll-margin-top:16px}}</style>
<h1>V163 · 공개 학습용 자료의 부위와 query를 분리해 보기</h1>
<div class="notice"><b>공통 40사례 판독은 완료. 전체390 판독 집계는 진행 중입니다.</b><p>기존 D/E 라벨·중심 소유 라벨·모델 예측을 가리고 원본 영상과 원래 crop을 확인했습니다. ROOT는 다른 판독 결과를 읽기 전에 자신의 40개 기록을 확정했습니다.</p><p><b>같은 계열 assistant 4회 판독의 약관찰입니다.</b> 사람 정답·독립 annotator gold·모델 정확도로 해석하지 않습니다. 불일치와 UNKNOWN은 그대로 보존하고 다수결로 정답을 만들지 않습니다.</p></div>
<p>40개 중 부위 가시성 관찰이 모두 같은 사례는25개, 다른 사례는15개입니다. 13개에서는 <b>부위 보임과 query 배경</b>이 동시에 일치했고,1개에서는 부위는 보이지만 query가 대상 팔레트의 다른 부위를 가리킨다는 관찰이 일치했습니다. 이는 native 균형 표본의 관찰이며 전체 발생률 추정이 아닙니다.</p>
<img src="{O / 'CROSS40_AGREEMENT_AND_QUERY_RELATION.png'}" alt="native별 가시성 판독 일치와 query 관계 차이">
<p>이 단계에서 새 모델 학습·채택은 없으며 초기 q/L/좌표, PnP, 기존 라벨을 바꾸지 않았습니다. 공개 TRAIN만 사용했고 사용자 평가223장과 공개 held45장은 이번 판독에 포함하지 않았습니다.</p>
<p>설명용 사례: <a href="#Q1608">부위 보임·중심 배경</a> · <a href="#Q0094">다른 목재 부위</a> · <a href="#Q1213">외부 차폐</a> · <a href="#Q1026">판독 불일치</a>. 아래 사례는 관찰 차이를 설명하려고 사후 선정한 예시이며 성능 표본이 아닙니다.</p>
{''.join(cards)}
<article><h2>고정40 전체</h2><table><thead><tr><th>사례 원본</th><th>native</th><th>부위 가시성</th><th>query 관계</th></tr></thead><tbody>{table}</tbody></table></article>
<p class="small">명목 CAD/pose 투영은 불확실한 부위 대응 참고 화면입니다. 실제 코너 픽셀 정답·3px 위치 품질·자기 가림 정답으로 쓰지 않습니다. 현재 공개 자료는 Euro WOOD만 있으며 SQUARE와 다른 재질의 실사 부위 감독이 누락되어 있습니다. native6도2개뿐입니다. 원래 WOOD/SQUARE/native8 목표는 유지합니다.</p>
<p><a href="{O / 'CROSS40_AGREEMENT_BEFORE_HISTORICAL_LABEL_JOIN.json'}">수치·판독 파일 연결</a> · <a href="{V / 'ROOT_ACTUAL_BLINDED_RGB_REVIEW_TRIGGER.json'}">고정 범위와 실제 판독 시작 기록</a></p></html>'''
dest = O / 'CROSS40_PREVIEW_KO.html'
assert not dest.exists()
dest.write_text(document)
print(dest)
