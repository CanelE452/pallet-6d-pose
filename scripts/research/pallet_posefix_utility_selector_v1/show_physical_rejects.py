"""All 17 additional physical-shape rejections, with unchanged source pixels."""
import base64
import html
import json

import numpy as np

from . import physical_shape_filter as S
from .list_pck20_failures import line
from scripts.research.pallet_posefix_corner_gate_v1.visualize import verify_metric,limits

P=S.P
OUT=P.OUT/'physical_shape_filter'
REASONS={'self_crossing':'면 내부 교차','concave_or_folded':'오목한 꺾임',
         'collapsed_face_or_corner':'면/꼭짓점 붕괴','winding_mismatch_with_PnP':'PnP 대비 면 순서 뒤집힘'}


def polygon(q,color,extra=''):
    points=' '.join(f'{x},{y}' for x,y in q)
    return f'<polygon points="{points}" fill="none" stroke="{color}" stroke-width="2.5" vector-effect="non-scaling-stroke" {extra}/>'


def main():
    for b in P.read(S.DOC/'OUTPUTS_LOCK.json')['artifacts']:P.verify_binding(b)
    result=P.read(S.DOC/'RESULTS.json');P.verify_binding(result['source_metrics'])
    decisions=P.read(S.RAW/'DECISIONS.json');metrics=P.read(P.N.RAW/'PER_FRAME_METRICS.json')
    source=P.read(S.A.RAW/'ACCEPTED_UNCHANGED_PREDICTIONS.json')
    from scripts.research.pallet_posefix_large_error_v1 import evaluate as O
    _,_,records,groups,pe,targets,_=O.evaluation_inputs()
    contract=P.read(P.ROOT/'_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')
    body=[];manifest=[];counts={}
    for ds,mode,name in [('DEV72','DEV72','비초록'),('GREEN150','GREEN150_MANUAL','초록')]:
        rejected=[r for r in decisions[ds] if not r['keep']]
        pred={r['id']:r for r in source[ds]};meta={r['id']:r for r in records[ds]}
        mm={r['id']:r for r in metrics[mode]['POSEFIX_RAW']}
        counts[ds]=len(rejected)
        body.append(f'<section id="{ds}"><h2>{name} {len(rejected)}장</h2>')
        for i,d in enumerate(rejected,1):
            row=pred[d['id']];record=meta[d['id']];metric=mm[d['id']]
            P.verify_binding(row['image']);P.verify_binding(record['annotation'])
            # The parent evaluation record does not necessarily contain image dimensions.
            from PIL import Image
            path=P.ROOT/row['image']['path']
            with Image.open(path) as image:w,h=image.size
            gt,box,modes,kind=O.read_targets(ds,record,[h,w],pe,targets)
            valid=dict(modes)[mode][:8]
            q=np.asarray(O.top(row['prediction'])['keypoints_xy'],float)
            assert metric['matched']
            verify_metric(q,gt,valid,np.asarray(groups[kind]['permutations']),metric,[h,w])
            eligible=[hyp for hyp in d['hypotheses'] if hyp['loo_pass'] and hyp['depth_pass']]
            assert eligible and all(not hyp['passed'] for hyp in eligible)
            selected=min(eligible,key=lambda x:x['loo_score'])
            faces=[f for f in selected['shape']['faces'] if not f['passed']]
            assert faces
            reproj=np.asarray(selected['projected_xy'],float)
            overlays=[];pnp=[];highlight=set();reason_labels=[]
            for a,b in contract['edges']:overlays.append(line(q[a],q[b],'#ffe34f',1.5))
            for f in faces:
                idx=f['corners'];highlight.update(idx)
                overlays.append(polygon(q[idx],'#ff665f','fill-opacity="0"'))
                pnp.append(polygon(reproj[idx],'#53dcff','stroke-dasharray="6 4"'))
                reason_labels.append(f'{f["name"]} ({"-".join(map(str,idx))}): '+', '.join(REASONS[r] for r in f['reasons']))
            for n in range(8):
                x,y=q[n]
                overlays.append(f'<circle cx="{x}" cy="{y}" r="2.5" fill="#ffe34f"/>')
                if n in highlight:
                    overlays.append(f'<text x="{x+5}" y="{y-5}" font-size="11" font-weight="bold" fill="white" stroke="#101820" stroke-width="2.5" paint-order="stroke">P{n}</text>')
            for j in np.flatnonzero(valid):
                x,y=gt[j]
                if ds=='GREEN150':overlays.extend([line((x-4,y-4),(x+4,y+4),'#ff58e5'),line((x-4,y+4),(x+4,y-4),'#ff58e5')])
                else:overlays.append(f'<circle cx="{x}" cy="{y}" r="4" fill="none" stroke="white" stroke-width="1.6"/>')
            crop=limits(np.concatenate([q[:8],gt[:8][valid]]),[h,w],25,110)
            x0,x1,y0,y1=crop;cropbox=f'{x0} {y0} {x1-x0} {y1-y0}';fullbox=f'0 0 {w} {h}'
            uri='data:'+('image/png' if path.suffix.lower()=='.png' else 'image/jpeg')+';base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
            ident=f'{ds}_{i}';label=f'{i}/{len(rejected)} · {d["id"]}'
            bad=sum(e>20 for e in metric['errors'])
            quality=f'평가점 {len(metric["errors"])}개 · 최대 오차 {max(metric["errors"]):.2f}px · '+(f'20px 초과 {bad}점' if bad else '평가점 전부 20px 이내')
            svg=f'<svg id="svg_{ident}" viewBox="{cropbox}" data-crop="{cropbox}" data-full="{fullbox}" role="img" aria-label="{html.escape(label)}"><image width="{w}" height="{h}" href="{uri}"/><g class="overlay">'+''.join(overlays)+'<g class="pnp">'+''.join(pnp)+'</g></g></svg>'
            all_hyp=[]
            for hyp in d['hypotheses']:
                rr=[]
                for f in hyp.get('shape',{}).get('faces',[]):
                    if not f['passed']:rr.append(f['name']+': '+', '.join(REASONS[r] for r in f['reasons']))
                loo=hyp.get('loo_score')
                all_hyp.append(html.escape(f'{hyp["name"]}: LOO={loo}; 카메라 앞={hyp["depth_pass"]}; 면 통과={hyp["face_pass"]}; '+' / '.join(rr)))
            body.append(f'<figure id="{ident}"><figcaption><strong>{html.escape(label)}</strong><br>{html.escape(" / ".join(reason_labels))}<br><span class="quality">{html.escape(quality)}</span></figcaption>{svg}'
                f'<div class="controls"><button onclick="zoom(\'svg_{ident}\',this)">전체 사진 보기</button><span>표시 가설: {html.escape(selected["name"])} · LOO {selected["loo_score"]:.5f}</span></div>'
                '<details><summary>검사한 모든 치수 가설</summary><p>'+'<br>'.join(all_hyp)+'</p></details></figure>')
            manifest.append(dict(dataset=ds,id=d['id'],image=row['image'],display_hypothesis=selected['name'],
                highlighted_faces=[f['name'] for f in faces],reasons=reason_labels,metric=metric,
                all_hypotheses_failed=True,source_decision=d))
        body.append('</section>')
    assert counts=={'DEV72':11,'GREEN150':6}
    page_text='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>추가 기하 필터 탈락 17장 · 비초록 11 / 초록 6</title><style>
*{box-sizing:border-box}body{margin:0;background:#101820;color:#eaf1f8;font:15px/1.55 system-ui,sans-serif}main{max-width:1050px;margin:auto;padding:16px}h1{font-size:25px}h2{margin-top:32px}nav{display:flex;gap:12px;flex-wrap:wrap;align-items:center}a{color:#9bd6ff}button{padding:8px 12px;border:1px solid #59748a;border-radius:5px;color:white;background:#263e51;cursor:pointer}figure{background:#172737;margin:18px 0 32px;border:1px solid #354c60}figcaption{padding:12px;overflow-wrap:anywhere}svg{display:block;width:100%;height:auto;max-height:850px}.quality{color:#abd5e9}.controls{padding:10px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:13px}details{padding:0 12px 12px;font-size:13px;overflow-wrap:anywhere}.pnp{display:none}body.showpnp .pnp{display:block}body.raw .overlay{display:none}.note{font-size:13px;color:#bfcedb}.warn{background:#342b20;padding:12px;border-left:3px solid #e8b45f}</style>
<main><header><h1>이번 기하 필터가 추가 탈락시킨 17장</h1><nav><a href="#DEV72">비초록 11장</a><a href="#GREEN150">초록 6장</a><button onclick="document.body.classList.toggle('raw');this.textContent=document.body.classList.contains('raw')?'판정 표시 켜기':'원본 사진만 보기'">원본 사진만 보기</button><button onclick="document.body.classList.toggle('showpnp');this.textContent=document.body.classList.contains('showpnp')?'PnP 면 숨기기':'PnP 면도 보기'">PnP 면도 보기</button></nav>
<p><b>빨간 테두리 = 탈락 원인 면</b> · 노란 선 = 변경하지 않은 Replay 보정 결과.<br>자홍색 × = 초록 수동 정답 · 흰색 ○ = 비초록 출처 미확인 기존 참조점. ‘PnP 면도 보기’를 누르면 하늘색 점선으로 비교할 투영 면이 나옵니다.</p>
<p class="warn">카메라 뒤쪽 점 때문에 탈락한 이미지는 0장입니다. 아래 17장은 모두 면 검사 탈락입니다. 그중 평가점이 모두 20px 이내인 이미지도 비초록 8장·초록 6장 포함돼 있습니다. ‘탈락’이 곧 실제 정답이 나빴다는 뜻은 아닙니다.</p>
<p class="note">각 사진은 한 번씩 나열했습니다. 기본은 팔레트 확대이며 아래 버튼으로 전체 사진을 볼 수 있습니다. 면 이름 near/far/top/right/bottom/left는 모델의 고정 코너 규약이며 화면상 좌우와 같지 않을 수 있습니다. 여러 치수 가설이 있을 때는 LOO를 통과한 물리적 가설 중 최소 LOO 가설을 대표로 표시하고, 나머지 실패 내역도 펼쳐 볼 수 있습니다.</p></header>'''+''.join(body)+'''</main><script>function zoom(id,button){const s=document.getElementById(id);const full=s.getAttribute('viewBox')===s.dataset.full;s.setAttribute('viewBox',full?s.dataset.crop:s.dataset.full);button.textContent=full?'전체 사진 보기':'팔레트 확대';}</script></html>'''
    OUT.mkdir(parents=True,exist_ok=True);page=OUT/'REJECTED_IMAGES.html';page.write_text(page_text,encoding='utf-8')
    P.freeze(OUT/'REJECTED_IMAGES_MANIFEST.json',dict(counts=counts,images=manifest,
        all_rejected_displayed=True,verified_metric_count=17,coordinates_changed=False,
        page=P.bound(page),source=P.bound(__file__),source_result=P.bound(S.DOC/'RESULTS.json'),
        source_decisions=P.bound(S.RAW/'DECISIONS.json')))
    print(json.dumps(dict(page=str(page),counts=counts),ensure_ascii=False))


if __name__=='__main__':main()
