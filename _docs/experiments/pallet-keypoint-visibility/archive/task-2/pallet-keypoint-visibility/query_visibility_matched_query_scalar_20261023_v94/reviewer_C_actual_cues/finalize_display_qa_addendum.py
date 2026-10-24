import json
import hashlib
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
O=P/'query_visibility_matched_query_scalar_20261023_v94/reviewer_C_actual_cues'
S=P/'query_visibility_flat_pallet_cue_bridge_20261023_v92/rgb96_review'
def bind(p):
    b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def write(p,x):
    assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    oldpath=O/'ALL160_C_ACTUAL_QUERY_CUE_OBSERVATIONS.jsonl'
    rows=[json.loads(x) for x in oldpath.read_text().splitlines()]
    changes=[]
    for fi,k,kind,note in [
        (2,3,'DECK_BLOCK_OUTER_TURN','동일 왼쪽 block/deck의 위쪽 외곽 turn이 crop 위쪽에 남는다. 최초 floor-turn 명칭은 표시 검토에서 부정확하여 정정했다. Query 중심과 단서는 떨어져 있다.'),
        (13,6,'FORK_OPENING_RIM','동일 pallet의 fork-opening 상단/측면 rim 구간이 보인다. 최초 block-junction 명칭보다 보수적으로 rim으로 기록하며 blur로 넓은 지역만 특정한다.'),
    ]:
        r=rows[fi*8+k]
        changes.append(dict(frame_index=fi,native_slot=k,old_kind=r['cue_kind'],new_kind=kind,old_notes=r['notes'],new_notes=note,geometry_state_and_ownership_changed=False))
        r['cue_kind']=kind;r['notes']=note
    for r in rows:
        b=r['crop_support_bbox_xyxy96_or_null'];u=r['support_uncertainty_px']
        raw=[b[0]-u,b[1]-u,b[2]+u,b[3]+u] if b else None
        r['support_uncertainty_bbox_unclipped96_or_null']=raw
        r['support_uncertainty_clipped_by_saved_crop']=bool(raw and any(x<0 or x>96 for x in raw))
    newpath=O/'ALL160_C_ACTUAL_QUERY_CUE_OBSERVATIONS_V2.jsonl';assert not newpath.exists()
    newpath.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows))
    source=[json.loads(x) for x in (S/'ROWS_INPUT_ONLY.jsonl').read_text().splitlines()]
    for r,s in zip(rows,source):
        assert r['input_id']==s['input_id'] and r['frame_index']==s['frame_index'] and r['native_slot']==s['native_slot']
        assert r['original_q']==s['query_xy'] and r['original_L']==s['L'] and r['query_center_valid']==s['query_center_valid']
        assert r['native_expected_part_state']=='UNKNOWN' and not r['training_admitted'] and not r['native_human_gold']
        assert not r['external_E_generated'] and not r['self_S_generated'] and not r['negative_training_label']
        if not r['query_center_valid']: assert r['visible_aux_structuralcue']=='UNKNOWN'
        if r['visible_aux_structuralcue']=='YES':
            assert r['query_center_valid'] and r['crop_support_bbox_xyxy96_or_null'] is not None
    v1summary=json.loads((O/'ACTUAL_CUE_REVIEW_SUMMARY_KO.json').read_text())
    a=np.load(S/'INPUT_RGB96_ONLY.npz',allow_pickle=False)['rgb96'].reshape(20,8,3,96,96)
    fdir=O/'raw_marked_saved96_gallery_v2';fdir.mkdir()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    gallery=list(v1summary['gallery'])
    for fi in [2,13]:
        canvas=Image.new('RGB',(1728,614),'#f4f4f4');d=ImageDraw.Draw(canvas)
        d.text((10,8),f'FRAME {fi:02d}: raw left / support+uncertainty right | V2 descriptive correction only | native UNKNOWN',fill='black',font=font)
        for k in range(8):
            x=(k%4)*432;y=42+(k//4)*286;r=rows[fi*8+k]
            raw=Image.fromarray(a[fi,k].transpose(1,2,0)).resize((192,192),Image.Resampling.NEAREST)
            canvas.paste(raw,(x+8,y+35));canvas.paste(raw,(x+224,y+35))
            d.text((x+8,y+2),f'k{k} aux={r["visible_aux_structuralcue"]} appearance={r["visible_pallet_structure_appearance"]}',fill='black',font=small)
            d.text((x+8,y+18),'selected owner '+r['target_ownership_confidence'][:35],fill='black',font=small)
            b=r['crop_support_bbox_xyxy96_or_null'];ub=r['support_uncertainty_bbox_xyxy96_or_null']
            if b:
                color='#00b050' if r['visible_aux_structuralcue']=='YES' else '#ff9500'
                d.rectangle([x+224+2*ub[0],y+35+2*ub[1],x+224+2*ub[2],y+35+2*ub[3]],outline='#f5d000',width=1)
                d.rectangle([x+224+2*b[0],y+35+2*b[1],x+224+2*b[2],y+35+2*b[3]],outline=color,width=2)
            d.text((x+8,y+235),(r['cue_kind'] or r['source_validity'])[:48],fill='#333333',font=small)
            d.text((x+8,y+253),'NO/UNKNOWN != external/self; training admitted false',fill='#555555',font=small)
        f=fdir/f'frame_{fi:02d}_raw_marked_v2.png';canvas.save(f);gallery[fi]=dict(frame_index=fi,figure=bind(f))
    add=dict(schema='V94_C_DISPLAY_QA_DESCRIPTIVE_CORRECTION_V2',base_observations=bind(oldpath),corrected_observations=bind(newpath),changes=changes,original_source_q_L_crop_bbox_uncertainty_and_decisions_unchanged=True,original_V1_files_preserved_in_place=True,
       marked_figures_actually_viewed_for_QA=[2,8,13,17],all20_raw_and_original_figures_actually_viewed=True,
       uncertainty_addition='Unclipped uncertainty and crop-boundary flag explicitly retained; no widening/relocation after model score. One ownership-U region extends beyond saved crop and remains U.',
       clipped_uncertainty_rows=sum(r['support_uncertainty_clipped_by_saved_crop'] for r in rows),no_new_model_current_fit_PnP=0)
    write(O/'DISPLAY_QA_DESCRIPTIVE_ADDENDUM_V2.json',add)
    summary=dict(v1summary);summary['observations']=bind(newpath);summary['gallery']=gallery;summary['display_QA_addendum']=bind(O/'DISPLAY_QA_DESCRIPTIVE_ADDENDUM_V2.json')
    summary['all160_q_L_order_and_validity_exact']=True
    summary['native_gold_and_training_admission']=False
    summary['support_bbox_is_not_expected_reference']='These are visible local structural support REGIONS. They are not a new native r, certified cue point, or expected native part association.'
    write(O/'ACTUAL_CUE_REVIEW_SUMMARY_V2_KO.json',summary)
    write(O/'FINAL_REVIEW_SEAL.json',dict(completed=True,PASS=True,root_scope=bind(P/'query_visibility_matched_query_scalar_20261023_v94/ROOT_PUBLIC_CUE_REVIEW_SCOPE_BEFORE_ACTUAL_CROPS.json'),summary=bind(O/'ACTUAL_CUE_REVIEW_SUMMARY_V2_KO.json'),observations=bind(newpath),raw_observation_seal=bind(O/'RAW_OBSERVATION_SEAL_BEFORE_MARKED_GALLERY.json'),display_addendum=bind(O/'DISPLAY_QA_DESCRIPTIVE_ADDENDUM_V2.json'),all20_figures_and_raw_sheets_viewed=bind(O/'ACTUAL_ALL20_FIGURE_AND_SAVED96_VIEW_RECEIPTS.json'),derived_gallery=gallery,code=[bind(O/'prepare_raw_saved96_sheets.py'),bind(O/'write_actual_observations.py'),bind(Path(__file__))],original_source_files_modified=0,new_model_forward_fit_private_PnP=0,training_admission=False,native_human_gold=False))
    print(json.dumps(dict(summary=bind(O/'ACTUAL_CUE_REVIEW_SUMMARY_V2_KO.json'),ledger=bind(newpath),seal=bind(O/'FINAL_REVIEW_SEAL.json'),valid_counts=summary['valid_aux_observations'],visible_structure=summary['valid_structure_appearance']),ensure_ascii=False))

if __name__=='__main__':main()
