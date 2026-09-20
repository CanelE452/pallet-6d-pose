"""Actual photos and fixed predictions: remaining manual-GT failures, not generated art."""
import base64
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import posthoc_loo as L
from scripts.research.pallet_posefix_corner_gate_v1.visualize import (
    BG, draw, limits, table, top, valid_points, verify_metric)

P = L.P
OUT = P.OUT / 'posthoc_loo'


def main():
    for lock_path in (L.DOC/'OUTPUTS_LOCK.json', P.DOC/'OUTPUTS_LOCK.json'):
        for b in P.read(lock_path)['artifacts']:
            P.verify_binding(b)
    result = P.read(L.DOC/'RESULTS.json')
    P.verify_binding(result['metrics'])
    original = {r['id']:r for r in P.read(P.RAW/'PREDICTIONS.json')['GREEN150']}
    outputs = {r['id']:r for r in P.read(L.RAW/'PREDICTIONS.json')['GREEN150']}
    decisions = {r['id']:r for r in P.read(L.RAW/'DECISIONS.json')['GREEN150']}
    utilities = {r['id']:r for r in P.read(P.RAW/'DECISIONS.json')['GREEN150']}
    old_metrics = P.read(P.RAW/'PER_FRAME_METRICS.json')['GREEN150_MANUAL']
    old_metrics = {a:{r['id']:r for r in rr} for a,rr in old_metrics.items()}
    metrics = {r['id']:r for r in P.read(L.RAW/'PER_FRAME_METRICS.json')['GREEN150_MANUAL']}
    contract = P.read(P.ROOT/'_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')
    permutations = np.asarray(next(r['permutations'] for r in contract['objects'] if r['object_type']=='plastic_standard_110x110x15'),int)
    failures = []
    for ident,new in metrics.items():
        before = old_metrics['A_N2'][ident]
        for j,v in enumerate(before['canonical_valid']):
            a,b = before['canonical_errors'][j],new['canonical_errors'][j]
            if v and a <= 10 < b:
                assert before['branch'] == new['branch'] and new['matched']
                native = int(np.flatnonzero(permutations[new['branch']] == j)[0])
                pair = next(i for i,p in enumerate(L.G.PAIRS) if native in p)
                assert decisions[ident]['pair_accept'][pair]
                failures.append(dict(id=ident, canonical=j, native=native, pair=pair,
                    before=a, after=b, increase=b-a))
    failures.sort(key=lambda r:(-r['increase'],r['id'],r['canonical']))
    assert len(failures)==15
    selected,seen = [],set()
    for r in failures:
        if r['id'] not in seen:
            selected.append(r); seen.add(r['id'])
        if len(selected)==3:
            break
    OUT.mkdir(parents=True,exist_ok=True)
    cards,manifest = [],[]
    names = ['Current N2','Learned selector','Learned selector + LOO']
    for rank,item in enumerate(selected,1):
        ident,j,native = item['id'],item['canonical'],item['native']
        row,diag = original[ident],decisions[ident]
        for key in ('image','annotation'):
            P.verify_binding(row[key])
        rgb = np.asarray(Image.open(P.ROOT/row['image']['path']).convert('RGB'))
        entries = P.read(P.ROOT/row['annotation']['path'])['objects'][0]['keypoint_annotations']
        gt = np.asarray([r['xy'] if r.get('xy') is not None else [np.nan,np.nan] for r in entries],float)
        valid = np.array([r.get('source')=='manual_click' and r.get('visibility',0)!=0 for r in entries]) & valid_points(gt)
        scores = [old_metrics['A_N2'][ident],old_metrics['LEARNED_PAIR'][ident],metrics[ident]]
        points = [np.asarray(top(row['predictions'][a])['keypoints_xy'],float) for a in ('A_N2','LEARNED_PAIR')]
        points.append(np.asarray(top(outputs[ident]['prediction'])['keypoints_xy'],float))
        for q,metric in zip(points,scores):
            np.testing.assert_array_equal(valid[:8],metric['canonical_valid'])
            verify_metric(q,gt,valid[:8],permutations,metric,row['raw_hw'])
        extent = limits(np.concatenate([gt[:8][valid[:8]],*[q[:8] for q in points]]),row['raw_hw'])
        zoom = limits([gt[j],*[q[native] for q in points]],row['raw_hw'],20,95)
        accepted=[]
        for mask in (diag['learned_pair_accept'],diag['pair_accept']):
            flags=[False]*8
            for keep,pair in zip(mask,L.G.PAIRS):
                for n in pair:flags[n]=keep
            accepted.append(flags)
        fig,axes=plt.subplots(2,3,figsize=(15,9),facecolor=BG)
        for col,(q,score) in enumerate(zip(points,scores)):
            for view,bounds in enumerate((extent,zoom)):
                draw(axes[view,col],rgb,q,points[0],gt,valid[:8],contract['edges'],bounds,
                     native,j,True,None if col==0 else accepted[col-1])
                axes[view,col].set_title(f'{names[col]}\nP{native} / G{j}: {score["canonical_errors"][j]:.2f} px',color='white',fontsize=12)
        pair=L.G.PAIRS[item['pair']]
        initial=diag['geometry']['per_corner_remove']
        final=diag['recheck_history'][-1]['per_corner_remove']
        assert not any(diag['recheck_history'][-1]['pair_rejected'])
        assert all(final[n] is not None and final[n]<=.05 for n in pair)
        utility=utilities[ident]['utility']
        fig.suptitle(f'Accepted by learned selector AND LOO, but worse than N2\n{ident}',color='white',fontsize=14,y=.985)
        footer=(f'Focus P{native} / G{j}: {item["before"]:.2f} -> {item["after"]:.2f} px. '
            f'Accepted pair P{pair[0]}-P{pair[1]}: final LOO {final[pair[0]]:.5f}, {final[pair[1]]:.5f} <= 0.05.\n'
            'Magenta x = manual GT; yellow lines = output; lime point = accepted correction; cyan point = N2 fallback.\n'
            'White arrows = correction from N2. Same RGB and crop in each column. GT used ONLY for evaluation/display.')
        fig.text(.5,.018,footer,ha='center',va='bottom',color='white',fontsize=9)
        fig.subplots_adjust(left=.01,right=.99,top=.875,bottom=.125,hspace=.23,wspace=.025)
        path=OUT/f'remaining_failure_{rank:02d}.png'
        fig.savefig(path,dpi=125,facecolor=BG);plt.close(fig)
        details=dict(**item,image=P.bound(path),pair_native=list(pair),
            pair_utility=[utility[n] for n in pair],initial_LOO=[initial[n] for n in pair],
            final_LOO=[final[n] for n in pair],GT_xy=gt[j].tolist(),
            coordinates=[q[native].tolist() for q in points],three_errors=[s['canonical_errors'][j] for s in scores])
        manifest.append(details)
        uri='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
        cards.append(f'<section id="case{rank}"><h2>{rank}. {item["before"]:.2f} → {item["after"]:.2f}px: 통과했지만 악화</h2>'
            f'<p class="mono">{html.escape(ident)} · 모델 P{native} ↔ 수동 정답 G{j}</p>'
            f'<a href="{path.name}" target="_blank"><img src="{uri}" alt="사례 {rank}: 기존 모델, 학습 선택기, 선택기와 LOO 비교"></a>'
            +table(['코너','예측 utility','첫 LOO','최종 LOO','판정'],[[f'P{n}',f'{utility[n]:+.4f}',f'{initial[n]:.5f}',f'{final[n]:.5f}','통과'] for n in pair])
            +'<p>Utility는 확률이 아닌 예측 이득입니다. LOO는 0.05 이하라 통과했지만 수동 정답과의 오차는 증가했습니다. 그림을 클릭하면 원본 크기로 열립니다.</p></section>')
    document='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>선택기 + LOO 통과 후 남은 오보정</title><style>*{box-sizing:border-box}body{margin:0;background:#0b131d;color:#eaf0f6;font-family:system-ui,sans-serif;line-height:1.6}main{max-width:1500px;margin:auto;padding:22px}section{background:#142130;border:1px solid #334a60;border-radius:12px;padding:20px;margin:22px 0}img{width:100%;height:auto}a{color:#8ed0ff}.mono{overflow-wrap:anywhere;font-family:monospace}table{border-collapse:collapse;width:100%;white-space:nowrap}td,th{padding:9px;border-bottom:1px solid #3a4d60;text-align:left}.tablewrap{overflow:auto}nav{display:flex;gap:20px;flex-wrap:wrap}.warn{border-left:4px solid #ffb259;padding:12px;background:#382b21}@media(max-width:650px){main{padding:10px}section{padding:12px}}</style><main>
<h1>선택기와 LOO를 모두 통과했는데도 나빠진 보정</h1>
<p class="warn">초록 수동 정답 기준, N2에서 10px 이내였지만 보정 후 10px 밖으로 나빠진 점이 <b>15개 / 11장</b> 남았습니다. 아래는 오차 증가량이 큰 순서로 서로 다른 사진 3장을 고른 실패 진단입니다. 전체 성능의 대표 표본은 아닙니다.</p>
<p>왼쪽: 기존 N2 · 가운데: 학습 선택기 · 오른쪽: 선택기 + 코너별 LOO. 위: 팔레트 전체, 아래: 같은 코너 확대. <b>자홍색 ×가 수동 정답</b>입니다. 평가 GT는 필터 판단에 넣지 않았습니다.</p>
<nav><a href="#case1">가장 큰 악화</a><a href="#case2">두 번째 사진</a><a href="#case3">세 번째 사진</a><a href="#all">남은 15점 전체</a></nav>'''+''.join(cards)
    document+='<section id="all"><h2>남은 손상 15점 전체</h2>'+table(['이미지','정답/모델 코너','N2 오차','선택기 + LOO 오차','증가량'],[[r['id'],f'G{r["canonical"]} / P{r["native"]}',f'{r["before"]:.2f}',f'{r["after"]:.2f}',f'+{r["increase"]:.2f}'] for r in failures])+'''</section>
<p>LOO는 다른 예측 점들과 기하적으로 일관되는지 검사할 뿐 실제 이미지 정답과의 일치를 보장하지 않습니다. 기존 카메라 K·등록 치수를 그대로 사용한 진단이며 카메라 메타데이터 불일치 이력도 있습니다. 이 세 이미지의 실패 원인이 그것이라고 단정하지 않습니다. 모델·정답·수도레이블은 변경하지 않았습니다.</p></main></html>'''
    (OUT/'REMAINING_FAILURES.html').write_text(document,encoding='utf-8')
    P.freeze(OUT/'FAILURE_VISUALIZATION.json',dict(selection='GREEN manual N2<=10px and learned+LOO>10px; descending error increase; top3 distinct images',
        all_failures=failures,figures=manifest,coordinate_metric_checks=9,
        source_metrics=P.bound(L.RAW/'PER_FRAME_METRICS.json'),html=P.bound(OUT/'REMAINING_FAILURES.html'),
        script=P.bound(__file__)))
    print(json.dumps(dict(figures=manifest,html=str(OUT/'REMAINING_FAILURES.html')),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
