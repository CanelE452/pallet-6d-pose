"""Post-completion ALL_FAILED2 diagnostic appendix, separate from fixed12.
Authorized by root after observing both retained failures; never replaces cases.
"""
from pathlib import Path
import argparse, importlib.util, json
import numpy as np
from PIL import Image,ImageDraw
import render_fixed12 as r

EXPECTED=['V161_A01_R2','V161_A38_R2']
TRIGGER_SHA='3e863ce18c04cf47b425ce029e3eaae568c48694948d7064ec1a96940530fdc6'
OUT=r.V/'all_failed2_visual_appendix_v1'
def main(trigger):
    assert r.sha(trigger)==TRIGGER_SHA
    t=r.read(trigger);assert t['builder_completed'] and t['approved_generated_source_readout'] and t['source_only']
    complete=r.read(r.check(t['complete']));assert complete['completed'] and complete['all159_attempts_retained']
    proofs=[json.loads(s) for s in r.check(complete['proofs']).read_text().splitlines() if s]
    failed=[x for x in proofs if not x['PASS']];assert [x['pair_id'] for x in failed]==EXPECTED
    assignments=[json.loads(s) for s in r.check(complete['assignments']).read_text().splitlines() if s];index={x['pair_id']:x for x in assignments}
    assert not OUT.exists();OUT.mkdir();m=r.builder();op=m.operators();records=[];images=[]
    for j,p in enumerate(failed,1):
        a=index[p['pair_id']];case={**a,'selection_rank':j};payload={}
        assert [k for k,v in p['proof'].items() if not v]==['sham_some_selected_local_donor_support']
        for name in r.VARIANTS:
            with np.load(r.check(p['variants'][name]),allow_pickle=False) as z:payload[name]={k:z[k].copy() for k in z.files}
            z=payload[name];k=a['native_slot'];assert (z['native_visibility_target']==-1).all() and (z['point_quality_target']==-1).all()
            rebuilt,valid=op['crop'](z['full_RGB'],a['query_xy'],a['L'],96)
            assert r.exact(rebuilt.transpose(1,2,0).copy(),z['rgb96'][k]) and r.exact(valid[0],z['valid96'][k])
        image=r.card(case,p,payload,m);d=ImageDraw.Draw(image)
        d.rectangle((0,0,1800,62),fill='#f5f8fb');d.text((22,20),f"ALL_FAILED2 진단 부록 {j:02d} · {p['pair_id']} · 실패 그대로 보존",font=r.font(31),fill='#ad482b')
        d.rectangle((0,137,1800,220),fill='#f5f8fb')
        d.text((22,141),'사후 전체 실패2개 전수 표시 · 사전고정12와 별도 · 실패를 성공으로 대체하지 않음',font=r.font(24),fill='#ad482b')
        d.text((22,181),'유일 실패: sham_some_selected_local_donor_support=False · sham은 선택 RGB96 밖 · 이동/재시도 없음',font=r.font(24),fill='#ad482b')
        path=OUT/f'ALL_FAILED2_{j:02d}_{p["pair_id"]}.png';image.save(path);images.append(image)
        records.append(dict(pair_id=p['pair_id'],diagnostic_scope='POST_COMPLETION_ALL_FAILED2_CENSUS_SEPARATE_FROM_PRESELECTED12',failure_predicates=[k for k,v in p['proof'].items() if not v],proof_detail=p['proof_detail'],source_variants=p['variants'],card=r.bind(path),new_labels=False,training_admitted=False))
    page=Image.new('RGB',(1800,2900),'white');page.paste(images[0],(0,0));page.paste(images[1],(0,1450));contact=OUT/'ALL_FAILED2_CONTACT.png';page.save(contact)
    hp=OUT/'index.html';hp.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>ALL_FAILED2 diagnostic appendix</title><style>body{font:20px system-ui;max-width:1800px;margin:auto;padding:24px}img{width:100%;height:auto}p{line-height:1.7}</style><h1>ALL_FAILED2 · 사후 전수 실패 진단 부록</h1><p>사전고정12를 바꾸지 않은 별도 부록입니다. 전체159 시도 중 두 실패를 모두 표시합니다. 두 경우 모두 sham_some_selected_local_donor_support=False가 유일한 실패이며, 원래 배치와 모든 결과를 유지합니다. 새로운 native 가시성·좌표품질은 UNKNOWN, 학습 편입은 false입니다. 픽셀 증명은 native 가시성 해결이나 자연 광학 가림 정답이 아닙니다.</p>'+''.join('<img src="'+Path(x['card']['path']).name+'" alt="all retained failure diagnostic">' for x in records)+'</html>')
    manifest=OUT/'ALL_FAILED2_MANIFEST.json';r.write(manifest,dict(completed=True,source_generation_complete=t['complete'],root_trigger=r.bind(trigger),records=records,contact=r.bind(contact),HTML=r.bind(hp)))
    r.write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,diagnostic_count=2,all_failures_included=True,fixed12_selection_changed=False,root_trigger=r.bind(trigger),code=r.bind(__file__),fixed12_renderer=r.bind(r.__file__),source_generation_complete=t['complete'],manifest=r.bind(manifest),actual_visual_QA_pending=True,HTML_browser_rendered=False,new_labels=False,training_admitted=False))
    print(json.dumps(dict(completed=True,complete=r.bind(OUT/'COMPLETE.json'))),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--trigger',required=True,type=Path);a=ap.parse_args();main(a.trigger)
