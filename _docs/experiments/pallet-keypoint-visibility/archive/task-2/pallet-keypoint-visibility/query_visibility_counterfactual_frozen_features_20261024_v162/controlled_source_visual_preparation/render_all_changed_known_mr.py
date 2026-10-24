"""Root-authorized exhaustive changed-known-MR appendix; not fixed selection."""
from pathlib import Path
import json
import numpy as np
from PIL import ImageDraw
import render_saved_C43_C44_caption_v2 as r

V=r.V;OLD=r.P/'query_visibility_mr6d_full_public_source_20261024_v153'
OUT=V/'controlled_source_visual_all_changed_mr_v1'
def main():
    trigger=V/'ROOT_BOTH_MODELS_COMPLETED_TRIGGER.json';assert r.sha(trigger)=='0953e23600c5e72ead259b717fe2053c385be16d845dfacf7b4166a8630eddf6'
    t=r.read(trigger);assert t['all2_completed'] and t['approved_saved_output_readout']
    censuspath=V/'source_result_readout/MR_HELD_KNOWN_DE_CHANGE_CENSUS.json';assert r.sha(censuspath)=='6b6924c3996462af9b8277f2790d02da6c464a1818e6aaaf7563f1ff112ef718'
    census=r.read(censuspath);changes=[json.loads(s) for s in r.check(census['change_rows']).read_text().splitlines() if s]
    assert census['changed_predictions']==len(changes)==1 and census['no_fixed12_replacement']
    framespath=OLD/'initializer_feature_preparation/FEATURE_FRAMES230_INPUT_ONLY.json';frames=r.read(framespath)['frames']
    factsfile=OLD/'full230_native_DINO_v1/FEATURE_INPUT_FACTS1840_INPUT_ONLY.jsonl';facts=[json.loads(s) for s in factsfile.read_text().splitlines() if s]
    cropsfile=OLD/'full230_native_DINO_v1/CROP_PACKETS230_INPUT_ONLY.json';crops={x['frame_index']:x['crop_packet'] for x in r.read(cropsfile)['sheets']}
    entry=r.read(r.H/'AGENT_RENDER_ENTRY_BOUND_TO_ROOT_TRIGGER_CAPTION_V2.json');packets={}
    for arm in r.ARMS:
        b=entry['prediction_packets'][arm]['MR1840']
        with np.load(r.check(b),allow_pickle=False) as z:packets[arm]={k:z[k].copy() for k in ('global_index','query_id','native_slot','query_xy','L','query_available','raw_three_logits','raw_owner_three_logits','owner_three_probabilities')}
    assert not OUT.exists();OUT.mkdir();records=[]
    for j,row in enumerate(changes,1):
        f=facts[row['global_index']];fr=frames[row['frame_index']]
        assert f['global_index']==row['global_index'] and f['query_id']==row['query_id'] and f['query_xy']==row['query_xy'] and f['L']==row['L'] and fr['split']=='SOURCE_VALIDATION'
        rgb=np.load(r.check(fr['canonical_RGB_NPY']),allow_pickle=False);assert r.ash(rgb)==fr['decoded_RGB_sha256']
        with np.load(r.check(crops[row['frame_index']]),allow_pickle=False) as z:
            ii=np.flatnonzero(z['global_index']==row['global_index']);assert len(ii)==1;crop=z['rgb96'][int(ii[0])].copy();valid=z['valid96'][int(ii[0])].copy()
        assert r.ash(crop)==f['RGB96_HWC_sha256'] and r.ash(valid)==f['valid96_sha256']
        ms=[]
        for arm,short in zip(r.ARMS,('C43','C44')):
            a=packets[arm];ii=np.flatnonzero(a['global_index']==row['global_index']);assert len(ii)==1;i=int(ii[0]);assert a['query_id'][i]==row['query_id'] and a['query_xy'][i].tolist()==row['query_xy'] and float(a['L'][i])==row['L']
            m=r.measure(a,i);assert m['primary_call']==row[short]['prediction_D_E_U'] and m['raw_external_E_logit']==row[short]['raw_visibility_E_logit'];ms.append(m)
        case={**row,'card_index':j,'family':'MR1840','weak_role_state':row['fixed_existing_weak_role_state']}
        image=r.old_card(case,rgb,crop,ms);d=ImageDraw.Draw(image)
        d.rectangle((0,0,1799,62),fill='#f5f8fb');d.text((24,22),f"ALL_CHANGED_KNOWN_MR_{j:02d} · native{row['native_slot']} · 기존 약역할 {row['fixed_existing_weak_role_state']}",font=r.font(31),fill='#20364c')
        d.rectangle((0,990,1799,1059),fill='#f5f8fb');d.text((24,1000),'사후 알려진 held D/E 변경 전수1/154 · 사전고정12와 별도 · 대표/성공사례 선택 주장 없음 · 새 라벨/학습 편입 없음',font=r.font(23),fill='#965111')
        path=OUT/f'ALL_CHANGED_KNOWN_MR_{j:02d}.png';image.save(path);records.append(dict(global_index=row['global_index'],query_id=row['query_id'],postresult_census_not_preselected_fixed12=True,original_weak_role=row['fixed_existing_weak_role_state'],measurements=dict(zip(r.ARMS,ms)),card=r.bind(path),new_labels=False,training_admitted=False))
    hp=OUT/'index.html';hp.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>All changed known MR</title><style>body{font:20px system-ui;max-width:1800px;margin:auto;padding:24px}img{width:100%;height:auto}</style><h1>ALL_CHANGED_KNOWN_MR · 사후 변경 전수 진단</h1><p>알려진 held D/E154개 중 바뀐 한 사례를 모두 표시합니다. 사전고정12를 대체하지 않으며 대표성이나 전체 성공을 주장하지 않습니다. 기존 E는 이전 assistant/root 약역할로 사람 gold가 아닙니다. q/L와 logit0 문턱은 그대로이며 이 readout은 새 라벨·학습 편입·모델 선택을 만들지 않습니다.</p>'+''.join('<img src="'+Path(x['card']['path']).name+'" alt="all changed known MR">' for x in records)+'</html>')
    manifest=OUT/'MANIFEST.json';r.write(manifest,dict(completed=True,root_trigger=r.bind(trigger),census=r.bind(censuspath),change_rows=census['change_rows'],input_metadata=[r.bind(framespath),r.bind(factsfile),r.bind(cropsfile)],records=records,HTML=r.bind(hp)))
    r.write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=r.bind(__file__),census=r.bind(censuspath),manifest=r.bind(manifest),all_changed_known_MR_count=1,original_known_scope=154,fixed12_selection_changed=False,new_labels=False,training_admitted=False,HTML_browser_rendered=False,actual_visual_QA_pending=True))
    base=V/'controlled_source_visual_v2/index.html';gallery=V/'controlled_source_visual_v2/index_with_all_changed_appendix.html'
    text=base.read_text().replace('<h1>C43/C44 · 고정 SOURCE/CF 설명 사례</h1>','<h1>C43/C44 · 고정 SOURCE/CF 설명 사례</h1><p><a href="../controlled_source_visual_all_changed_mr_v1/index.html">별도 사후 ALL_CHANGED_KNOWN_MR 전수 진단1/154 보기</a></p><p>CF 카드 상단은 V161 생성 당시 증명 기록이며 하단은 V162 실제 학습 후 저장 출력입니다. C43은 CF0, C44는 CF314 owner 보조로 학습했습니다.</p>')
    gallery.write_text(text);print(json.dumps(dict(appendix_complete=r.bind(OUT/'COMPLETE.json'),final_gallery=r.bind(gallery))),flush=True)
if __name__=='__main__':main()
