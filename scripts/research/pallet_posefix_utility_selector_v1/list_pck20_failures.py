"""One vertical image stream: every retained Replay image failing PCK20."""
import base64
import html
import json

import numpy as np

from . import all8_replay_frame_filter as A
from scripts.research.pallet_posefix_corner_gate_v1.visualize import verify_metric

P=A.P
OUT=P.OUT/'all8_replay_frame_filter'


def line(a,b,color,width=2,extra=''):
    return f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{color}" stroke-width="{width}" vector-effect="non-scaling-stroke" {extra}/>'


def rect(box,color):
    x,y,r,b=box
    return f'<rect x="{x}" y="{y}" width="{r-x}" height="{b-y}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="7 5" vector-effect="non-scaling-stroke"/>'


def main():
    for binding in P.read(A.DOC/'OUTPUTS_LOCK.json')['artifacts']:P.verify_binding(binding)
    results=P.read(A.DOC/'RESULTS.json');P.verify_binding(results['metrics_source'])
    decisions=P.read(A.RAW/'DECISIONS.json')
    metrics=P.read(P.N.RAW/'PER_FRAME_METRICS.json')
    saved=P.read(P.N.RAW/'PREDICTIONS.json')
    from scripts.research.pallet_posefix_large_error_v1 import evaluate as O
    _,_,records,groups,pe,targets,_=O.evaluation_inputs()
    contract=P.read(P.ROOT/'_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')
    sections=[];manifest=[];counts={}
    for ds,mode,label in [('DEV72','DEV72','비초록'),('GREEN150','GREEN150_MANUAL','초록')]:
        keep={r['id'] for r in decisions[ds] if r['keep']}
        by_id={r['id']:r for r in metrics[mode]['POSEFIX_RAW']}
        chosen=[r for r in saved[ds] if r['id'] in keep and any(e>20 for e in by_id[r['id']]['errors'])]
        meta={r['id']:r for r in records[ds]}
        counts[ds]=dict(images=len(chosen),unmatched=sum(not by_id[r['id']]['matched'] for r in chosen),
            failed_corners=sum(e>20 for r in chosen for e in by_id[r['id']]['errors']))
        title=f'{label} · {len(chosen)}장'
        sections.append(f'<section id="{ds}"><h2>{title}</h2>')
        for i,row in enumerate(chosen,1):
            m=by_id[row['id']]
            for k in ('image','annotation'):P.verify_binding(row[k])
            gt,box,modes,kind=O.read_targets(ds,meta[row['id']],row['raw_hw'],pe,targets)
            valid=dict(modes)[mode][:8]
            assert np.array_equal(valid,m['canonical_valid'])
            pred=O.top(row['predictions']['POSEFIX_RAW']);q=np.asarray(pred['keypoints_xy'],float)
            perm=np.asarray(groups[kind]['permutations'][m['branch']],int)
            if m['matched']:
                verify_metric(q,gt,valid,np.asarray(groups[kind]['permutations']),m,row['raw_hw'])
            h,w=row['raw_hw'];path=P.ROOT/row['image']['path']
            mime='image/png' if path.suffix.lower()=='.png' else 'image/jpeg'
            uri=f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
            finite=np.isfinite(q).all(1)&~(q==-1).all(1)
            overlays=[]
            for a,b in contract['edges']:
                if finite[a] and finite[b]:overlays.append(line(q[a],q[b],'#ffe34f',1.7))
            for n in range(8):
                if finite[n]:overlays.append(f'<circle cx="{q[n,0]}" cy="{q[n,1]}" r="2.8" fill="#ffe34f"/>')
            for j in np.flatnonzero(valid):
                x,y=gt[j]
                if ds=='GREEN150':
                    overlays.extend([line((x-4,y-4),(x+4,y+4),'#ff58e5'),line((x-4,y+4),(x+4,y-4),'#ff58e5')])
                else:overlays.append(f'<circle cx="{x}" cy="{y}" r="4" fill="none" stroke="#eeeeee" stroke-width="2"/>')
                if m['matched'] and m['canonical_errors'][j]>20:
                    n=int(np.flatnonzero(perm==j)[0])
                    overlays.append(line(q[n],gt[j],'#ff665f',1.3,'stroke-dasharray="4 3"'))
                    overlays.append(f'<circle cx="{q[n,0]}" cy="{q[n,1]}" r="6" fill="none" stroke="#ff665f" stroke-width="2"/>')
            failed=sum(e>20 for e in m['errors'])
            if not m['matched']:
                overlays.extend([rect(pred['box_xyxy'],'#ffad42'),rect(box,'#ffffff')])
                status='박스 매칭 실패 — 표시된 벌점은 실제 코너 거리 아님'
            else:status=f'20px 초과 {failed}점 · 최대 {max(m["errors"]):.1f}px'
            caption=f'{i}/{len(chosen)} · {row["id"]} · {status}'
            svg=(f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(caption)}">'
                 f'<image width="{w}" height="{h}" href="{uri}"/>'
                 '<g class="overlay">'+''.join(overlays)+'</g></svg>')
            sections.append(f'<figure id="{ds}_{i}">{svg}<figcaption>{html.escape(caption)}</figcaption></figure>')
            manifest.append(dict(dataset=ds,id=row['id'],image=row['image'],matched=m['matched'],
                failed_corners=failed,canonical_errors=m['canonical_errors'],branch=m['branch']))
        sections.append('</section>')
    assert counts['DEV72']==dict(images=23,unmatched=0,failed_corners=64)
    assert counts['GREEN150']==dict(images=13,unmatched=11,failed_corners=53)
    document='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>20px 실패 이미지 전체 · 비초록 23 / 초록 13</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101820;color:#edf3f9;font:15px/1.5 system-ui,sans-serif}main{max-width:1050px;margin:auto;padding:14px}header{padding:10px 0 20px}h1{font-size:24px}h2{font-size:22px;margin:28px 0 12px}nav{display:flex;gap:15px;align-items:center;flex-wrap:wrap}a{color:#a0d7ff}button{padding:8px 12px;background:#223c50;color:white;border:1px solid #56728a;border-radius:6px;cursor:pointer}figure{margin:0 0 30px;background:#172534}svg{width:100%;height:auto;display:block}figcaption{padding:9px;overflow-wrap:anywhere;font-size:13px}body.raw .overlay{display:none}.note{color:#b8c7d4;font-size:13px}section{scroll-margin-top:8px}</style>
<main><header><h1>20px 기준 실패 이미지 — 한 장씩 전체 나열</h1><nav><a href="#DEV72">비초록 23장</a><a href="#GREEN150">초록 13장</a><button id="toggle" type="button" onclick="document.body.classList.toggle('raw');this.textContent=document.body.classList.contains('raw')?'보정·기준점 표시':'원본 사진만 보기'">원본 사진만 보기</button></nav>
<p>8점 LOO 통과 집합(비초록 55장·초록 124장)에서 평가 코너 하나라도 20px 밖이거나 박스 매칭이 실패한 사진입니다. 같은 사진은 한 번만 표시합니다.</p>
<p class="note">노란 선: 고정 Replay 보정 결과 · 빨간 원/점선: 20px 초과 코너 · 초록의 자홍색 ×: 수동 정답 · 비초록의 흰색 ○: 출처 미확인 기존 참조점.<br>초록 13장 중 11장은 박스 매칭 실패입니다. 이때 주황 점선은 검출 박스, 흰 점선은 평가 박스이며 800px 실패 벌점을 실제 이동 거리처럼 표시하지 않았습니다. 나머지 초록 2장은 실제 코너 오차가 20px를 넘었습니다.</p></header>'''+''.join(sections)+'</main></html>'
    OUT.mkdir(parents=True,exist_ok=True)
    page=OUT/'PCK20_FAILURES_ALL.html';page.write_text(document,encoding='utf-8')
    P.freeze(OUT/'PCK20_FAILURES_MANIFEST.json',dict(counts=counts,frames=manifest,
        selection='kept by fixed all8 Replay median LOO; any full-penalty error>20; original dataset order; no omitted failure frames',
        page=P.bound(page),source=P.bound(__file__),source_metrics=results['metrics_source'],
        input_decisions=P.bound(A.RAW/'DECISIONS.json'),coordinates_changed=False))
    print(json.dumps(dict(page=str(page),counts=counts),ensure_ascii=False))


if __name__=='__main__':main()
