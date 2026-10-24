"""All accepted fixed-order public pairs; pinned same-q RGB96, no feature/model."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import protected_union_adapter as a
from normalize_and_lock import HERE, P, bind, read, lines, write


def main():
    src=HERE/'image_only_v2_exclusive_alpha';out=HERE/'image_only_v2_exclusive_alpha_QA'
    out.mkdir(exist_ok=False)
    rows=lines(HERE/'ALL960_NORMALIZED_INPUT_ONLY.jsonl');lineage=lines(HERE/'SOURCE960_DRIVER_LINEAGE.jsonl')
    f=lines(src/'FEASIBILITY960_GT_ONLY.jsonl');occ=lines(src/'OCCURRENCES2880_LINEAGE.jsonl')
    accepted=[x['base_row_index']for x in f if x['pair_ready']]
    assert accepted==sorted(accepted) and len(accepted)==15
    write(out/'SCOPE_BEFORE_RENDER.json',dict(completed=True,selection='ALL15 accepted V2 pairs, original global row order, first8 thenlast7; no model/source score ranking or replacement',
          selected_rows=accepted,V2_seal=bind(src/'IMAGE_ONLY_INTAKE_SEAL.json'),code=bind(__file__),
          same_original_q_L=True,three_raw96_plus_separate_support_overlay=True,
          original_input_P3_tensor_or_model_current_reads=0,conditional_AI_teacher_not_human_native_gold=True))
    crop,_=a.v3.v1.frozen_rgb96_operator(P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py','802f2e17f1ee98ae09d8468b8ab749653a50054a85f24ef0af16385a7c262b66')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    titlefont=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    panels=[];proofs=[];raw=out/'raw96';raw.mkdir()
    for i in accepted:
        row=rows[i];line=lineage[i];meta=f[i];q=np.asarray(row['query_xy'],np.float64);L=row['L'];rgb96=[];valid96=[];bindings=[]
        for vi in range(3):
            o=occ[3*i+vi]
            if vi==0:
                path=Path(line['fullRGB_path']);expected=line['fullRGB_file_binding']['sha256']
                rgb=np.load(path,allow_pickle=False)if line['fullRGB_format']=='CANONICAL_RGB_UINT8_HWC_NPY'else np.asarray(Image.open(path).convert('RGB')).copy()
            else:
                path=Path(o['edited_fullRGB_file']['path']);expected=o['edited_fullRGB_file']['sha256'];rgb=np.asarray(Image.open(path).convert('RGB')).copy()
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
            assert hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()==o['full_rgb_decoded_sha256']
            c,v=crop(rgb,q,L,96);rgb96.append(c);valid96.append(v)
            rp=raw/f'row_{i:04d}_{o["variant"]}_RAW96.png';Image.fromarray(c.transpose(1,2,0)).save(rp);bindings.append(bind(rp))
        assert np.array_equal(valid96[0],valid96[1])and np.array_equal(valid96[0],valid96[2])
        protected=np.load(meta['protected_union_file']['path'],allow_pickle=False)['protected_union']
        u=(np.arange(96,dtype=np.float64)+.5)/96-.5;xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L)
        xi,yi=np.floor(xx+.5).astype(int),np.floor(yy+.5).astype(int);inside=(xi>=0)&(xi<protected.shape[1])&(yi>=0)&(yi<protected.shape[0])
        mask=np.zeros((96,96),bool);mask[inside]=protected[yi[inside],xi[inside]]
        padded=np.pad(mask,1);erode=np.logical_and.reduce([padded[y:y+96,x:x+96]for y in range(3)for x in range(3)])
        boundary=mask&~erode;overlay=rgb96[0].transpose(1,2,0).copy();overlay[boundary]=[255,70,70]
        marked=Image.fromarray(overlay).resize((192,192),Image.Resampling.NEAREST);d=ImageDraw.Draw(marked)
        d.line((89,95,101,95),fill=(0,235,255),width=2);d.line((95,89,95,101),fill=(0,235,255),width=2)
        tile=Image.new('RGB',(824,252),'#101820');d=ImageDraw.Draw(tile)
        d.text((12,4),f'row{i:04d} | {line["source"]} {row["split"]} | logical slot{row["native_slot_logical_only"]} | L={L:.3f}',font=titlefont,fill='white')
        for col,lab in enumerate(['BEFORE raw','COVER raw','SHAM raw','Protected union + q']):
            x=12+col*202;d.text((x,29),lab,font=font,fill='#cfe0ee')
            image=Image.fromarray(rgb96[col].transpose(1,2,0)).resize((192,192),Image.Resampling.NEAREST)if col<3 else marked
            tile.paste(image,(x,49))
        panels.append(tile)
        npz=raw/f'row_{i:04d}_RGB96_PROOF.npz';np.savez_compressed(npz,rgb96=np.stack(rgb96),crop_valid96=np.stack(valid96),protected_nearest96_diagnostic_only=mask)
        proofs.append(dict(base_row_index=i,source=line['source'],split=row['split'],input_id=row['input_id'],logical_slot=row['native_slot_logical_only'],
                           unchanged_q=row['query_xy'],unchanged_L=L,pair_certificate=meta,raw96=bindings,RGB96_proof=bind(npz),
                           exact_pinned_crop_formula=True,fullRGB_SHA_verified=True,model_P3_current_fit_PnP=0))
    sheets=[]
    for page,start in enumerate(range(0,15,8),1):
        part=panels[start:start+8];sheet=Image.new('RGB',(824,80+252*len(part)),'#101820');d=ImageDraw.Draw(sheet)
        d.text((12,8),f'V95 V2 ALL15 accepted same-q pairs | page{page} | native UNKNOWN / AI auxiliary only',font=titlefont,fill='white')
        d.text((12,33),'Original q/L; red support union and cyan q are display-only. No reinitializer/P3/model/fit.',font=font,fill='#cfe0ee')
        for j,tile in enumerate(part):sheet.paste(tile,(0,80+j*252))
        path=out/f'CONTACT_{page:02d}_ALL15_FIXED_ORDER.png';sheet.save(path);sheets.append(bind(path))
    write(out/'ALL15_PAIR_CROP_PROVENANCE.json',dict(completed=True,pairs=proofs,sheets=sheets,root_actual_view_pending=True,
          raw_crop_modified=False,overlay_separate=True,semantic_gold=False,training_admission=False))
    write(out/'SEAL.json',dict(completed=True,PASS=True,V2_source_seal=bind(src/'IMAGE_ONLY_INTAKE_SEAL.json'),
          metadata=bind(out/'ALL15_PAIR_CROP_PROVENANCE.json'),scope=bind(out/'SCOPE_BEFORE_RENDER.json'),sheets=sheets,
          files=[bind(p)for p in sorted(raw.iterdir())],new_model_P3_weights_fit_current_PnP=0))
    print(json.dumps(dict(seal=bind(out/'SEAL.json'),sheets=sheets),indent=2))


if __name__=='__main__':main()
