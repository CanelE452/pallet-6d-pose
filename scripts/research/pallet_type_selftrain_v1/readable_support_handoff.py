"""Read-only provenance audit and easier RGB review proposal; never promote GT."""
from collections import Counter
from html import escape
import os
from . import common as C

DOC=C.DOC/'large_corner_recovery_v1/readable_support_handoff_v1'
OUT=C.OUT/'large_corner_recovery_v1/readable_support_handoff_v1'


def main():
    scout_path=C.OUT/'large_corner_recovery_v1/readability_scout/SCOUT.json'
    scout=C.read(scout_path)
    for binding in scout['sources']:C.verify(binding)
    selected=[r for r in scout['records'] if r['index'] in {7,10,11,12}]
    assert len(selected)==4
    for r in selected:C.verify(r['image'])
    counts={};sources=[]
    for name in ('gt_manual','gt_final'):
        paths=sorted((C.ROOT/'data/pallet/raw_data/capture0403middle'/name).glob('*.json'))
        types=Counter();provenance=0;clicks=0
        for p in paths:
            d=C.read(p);sources.append(C.bound(p))
            for obj in d.get('objects',[]):
                types[obj.get('gt_source','unspecified')]+=1
                entries=obj.get('keypoint_annotations',[])
                provenance+=bool(entries)
                clicks+=sum(x.get('source')=='manual_click' for x in entries)
        counts[name]=dict(files=len(paths),gt_source=dict(types),objects_with_point_provenance=provenance,explicit_manual_clicks=clicks)
    workspace_path=C.OUT/'large_corner_recovery_v1/real_support_annotations/WORKSPACE.json'
    workspace=C.read(workspace_path);saved=[]
    for r in workspace['rows']:
        p=C.ROOT/r['annotation']
        if p.exists():
            d=C.read(p)
            saved.append(dict(id=r['id'],role=r['role'],annotation=C.bound(p),
                manual_visible_corners=sum(x.get('source')=='manual_click' and x.get('visibility')==2
                    for obj in d.get('objects',[]) for x in obj.get('keypoint_annotations',[])[:8])))
    result=dict(status='EASIER_RGB_PROPOSAL_ONLY_REQUIRES_USER_ANNOTATION',
        saved=saved,legacy_audit=counts,legacy_sources=sources,selected=selected,
        caveats=['No coordinates generated or accepted as manual GT.',
            'capture0403middle shares indoor setting/date with eval_noapril: must audit acquisition and training history before any train/validation promotion.',
            'Four examples are a usability proposal, not a sufficient training/validation sample size.',
            'Original 30-frame annotation workspace and its saved image are untouched.',
            'Extreme-low evaluation exclusion does not fix large corner errors.'],
        sources=[C.bound(scout_path),C.bound(workspace_path),C.bound(__file__)])
    C.freeze(DOC/'AUDIT.json',result)
    cards=[]
    for r in selected:
        src=os.path.relpath(C.ROOT/r['image']['path'],OUT)
        cards.append(f'<figure><figcaption>{escape(r["session"])} / {escape(r["image"]["path"].split("/")[-1])}</figcaption><img src="{escape(src)}"></figure>')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>윗면이 보이는 어노테이션 후보</title><style>body{background:#15232e;color:white;font:18px sans-serif;margin:24px}img{width:640px;max-width:100%}figure{margin:24px 0}p{max-width:900px}</style><h1>윗면이 보이는 후보 4장</h1><p>극저각 대신 수동 코너 확인이 쉬운 사진 예시입니다. 아직 정답이나 학습 데이터가 아닙니다. 기존 작업 창과 저장한1장은 그대로 유지했습니다.</p><p>실내 촬영 장면의 평가 세션과 관계·기존 학습 이력은 실제 학습 편입 전에 추가 확인해야 합니다. 이4장만으로 큰 오차 복구가 검증되는 것은 아닙니다.</p>'+''.join(cards)+'</html>'
    C.write_text(OUT/'index.html',page)
    print(counts);print('Saved:',[(r['role'],r['manual_visible_corners']) for r in saved]);print(OUT/'index.html')


if __name__=='__main__':main()
